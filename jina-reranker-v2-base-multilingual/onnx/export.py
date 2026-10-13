"""Export jina-reranker-v2-base-multilingual to ONNX (fp32, single file - model is ~1.1 GB)."""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convert import MODEL_NAME, REVISION, load_native_model  # noqa: E402

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
final = OUT / "model_fp32.onnx"


class Exportable(torch.nn.Module):
    """Same math as XLMRobertaForSequenceClassification, but with the classic additive attention mask.

    transformers 5 builds the mask with Gather/And/Where ops that the ONNX Runtime transformer optimizer
    can't match, so attention would stay unfused. (1 - mask) * -10000 is the pattern it recognizes.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask):
        hidden = self.m.roberta.embeddings(input_ids=input_ids)
        ext = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(hidden.dtype)) * -10000.0
        hidden = self.m.roberta.encoder(hidden, attention_mask=ext).last_hidden_state
        return self.m.classifier(hidden)


tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
# eager attention exports explicit MatMul/Softmax ops that ORT fuses into a single Attention op
model = load_native_model(attn_implementation="eager")
wrapped = Exportable(model).eval()

enc = tok(["what is panda?"] * 2, ["The giant panda is a bear.", "hi there"], padding=True, return_tensors="pt")
with torch.inference_mode():
    ref = model(**enc).logits
    got = wrapped(enc["input_ids"], enc["attention_mask"])
    assert torch.allclose(ref, got, atol=1e-4), f"wrapper differs from HF model: {ref} vs {got}"
    torch.onnx.export(
        wrapped,
        (enc["input_ids"], enc["attention_mask"]),
        str(final),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
print("saved", final)
