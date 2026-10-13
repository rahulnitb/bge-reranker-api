"""Load jina-reranker-v2-base-multilingual as a standard Hugging Face XLMRobertaForSequenceClassification.

Jina ships the model with custom flash-attention code (trust_remote_code) written for transformers 4.x; it does
not import on transformers 5. The architecture itself is plain XLM-R base (post-norm, absolute positions, exact
GELU), only the weights are laid out differently (fused Wqkv, mixer/mlp/norm names). This maps them onto the
built-in class, so no remote code runs and the model exports/fuses like any XLM-R. Weights are stored in bf16;
casting to fp32 is exact.
"""
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"  # pinned: the files this mapping was verified against


def native_state_dict(src: dict) -> dict:
    out = {}
    for k, v in src.items():
        v = v.float()
        if k == "roberta.emb_ln.weight":
            out["roberta.embeddings.LayerNorm.weight"] = v
        elif k == "roberta.emb_ln.bias":
            out["roberta.embeddings.LayerNorm.bias"] = v
        elif k.startswith("roberta.encoder.layers."):
            i, rest = k[len("roberta.encoder.layers."):].split(".", 1)
            p = f"roberta.encoder.layer.{i}."
            if rest.startswith("mixer.Wqkv."):
                kind = rest.rsplit(".", 1)[1]  # weight | bias
                for name, part in zip(("query", "key", "value"), v.chunk(3, dim=0)):  # rows are [q; k; v]
                    out[f"{p}attention.self.{name}.{kind}"] = part.contiguous()
            else:
                rename = {"mixer.out_proj.": "attention.output.dense.", "norm1.": "attention.output.LayerNorm.",
                          "mlp.fc1.": "intermediate.dense.", "mlp.fc2.": "output.dense.", "norm2.": "output.LayerNorm."}
                pre = next(r for r in rename if rest.startswith(r))
                out[p + rename[pre] + rest[len(pre):]] = v
        else:  # embeddings.* and classifier.* already use the Hugging Face names
            out[k] = v
    return out


def load_native_model(attn_implementation: str = "sdpa") -> XLMRobertaForSequenceClassification:
    config = XLMRobertaConfig(
        vocab_size=250002, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
        hidden_act="gelu", max_position_embeddings=1026, type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1,
        bos_token_id=0, eos_token_id=2, num_labels=1, attn_implementation=attn_implementation,
    )
    model = XLMRobertaForSequenceClassification(config)
    weights = load_file(hf_hub_download(MODEL_NAME, "model.safetensors", revision=REVISION))
    missing, unexpected = model.load_state_dict(native_state_dict(weights), strict=False)
    # position_ids buffers are recomputed by the model; anything else missing/unexpected is a mapping bug
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"weight mapping mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()
