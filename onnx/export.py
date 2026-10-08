"""Export bge-reranker-v2-m3 to ONNX (fp32, weights in external data file)."""
import sys
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "BAAI/bge-reranker-v2-m3"
OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
raw = OUT / "raw" / "model.onnx"
raw.parent.mkdir(exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
model.config.return_dict = False

enc = tok(["what is panda?"] * 2, ["The giant panda is a bear.", "hi there"], padding=True, return_tensors="pt")
with torch.inference_mode():
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(raw),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
print("exported", raw)
# re-save with weights as external data (protobuf 2GB limit)
m = onnx.load(str(raw))
final = OUT / "model_fp32.onnx"
onnx.save(m, str(final), save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp32.onnx.data")
print("saved", final)
