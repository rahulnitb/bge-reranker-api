"""Export Qwen3-Reranker-0.6B to ONNX (fp32, weights in an external data file - ~2.4 GB > protobuf's 2 GB limit).

The exported graph is qwen_scoring.YesNoScorer: input_ids + attention_mask (left padded) -> [batch, 1] relevance
logit (logit "yes" - logit "no"). No KV cache, no full LM head.
"""
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_scoring import DEFAULT_INSTRUCTION, build_inputs, load_scorer, load_tokenizer  # noqa: E402

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
raw = OUT / "raw" / "model.onnx"
raw.parent.mkdir(exist_ok=True)
final = OUT / "model_fp32.onnx"

tok = load_tokenizer()
scorer = load_scorer(attn_implementation="eager")
ids, mask = build_inputs(tok, "what is panda?", ["The giant panda is a bear.", "hi"], DEFAULT_INSTRUCTION, 512)
ids, mask = torch.from_numpy(ids), torch.from_numpy(mask)
with torch.inference_mode():
    ref = scorer(ids, mask)
    torch.onnx.export(
        scorer,
        (ids, mask),
        str(raw),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
print("exported", raw, "reference logits", ref.view(-1).tolist())
m = onnx.load(str(raw))
onnx.save(m, str(final), save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32.onnx.data")
print("saved", final)
