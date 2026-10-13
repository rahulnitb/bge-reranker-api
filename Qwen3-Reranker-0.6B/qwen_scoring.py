"""Qwen3-Reranker-0.6B scoring, shared by the API, the ONNX export and the validation script.

Qwen3-Reranker is a causal LM, not a classifier. Each (query, document) pair is wrapped in a fixed chat prompt that
asks the model to answer "yes" or "no"; the relevance logit is logit("yes") - logit("no") at the last position, and
sigmoid of it equals the softmax probability of "yes" used by the official example. Only those two rows of the
151k-token LM head are computed.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"  # pinned: the files this code was verified against
YES_ID, NO_ID = 9693, 2152  # "yes" / "no" (1_LogitScore/config.json in the model repo)
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Prompt format from the model card / chat_template.jinja
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_tokenizer(path=MODEL_NAME):
    kw = {"revision": REVISION} if path == MODEL_NAME else {}
    return AutoTokenizer.from_pretrained(path, padding_side="left", **kw)


def build_inputs(tok, query, docs, instruction, max_length):
    """Left-padded int64 input_ids/attention_mask (numpy). max_length counts the whole prompt; when a pair is too
    long, only the end of the text is cut - the document comes last, so the query and the prompt stay intact."""
    pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
    bodies = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs]
    enc = tok(bodies, add_special_tokens=False, truncation=True, max_length=max_length - len(pre) - len(suf))
    ids = [pre + x + suf for x in enc["input_ids"]]
    width = max(len(x) for x in ids)
    input_ids = np.full((len(ids), width), tok.pad_token_id, dtype=np.int64)
    mask = np.zeros((len(ids), width), dtype=np.int64)
    for i, x in enumerate(ids):  # left padding: the last position is always the real last token
        input_ids[i, width - len(x):] = x
        mask[i, width - len(x):] = 1
    return input_ids, mask


class YesNoScorer(torch.nn.Module):
    """input_ids, attention_mask (left padded) -> [batch, 1] logit("yes") - logit("no") at the last position."""

    def __init__(self, causal_lm):
        super().__init__()
        self.body = causal_lm.model
        w = causal_lm.lm_head.weight  # tied with the input embeddings
        self.register_buffer("yes_minus_no", (w[YES_ID] - w[NO_ID]).detach().clone().unsqueeze(1))  # [hidden, 1]

    def forward(self, input_ids, attention_mask):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, -1, :] @ self.yes_minus_no


def load_scorer(attn_implementation="sdpa"):
    lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=REVISION, dtype=torch.float32,
                                              attn_implementation=attn_implementation)
    return YesNoScorer(lm.eval()).eval()
