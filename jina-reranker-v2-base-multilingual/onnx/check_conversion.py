"""Optional: prove convert.py is exact by comparing it with Jina's own remote code (needs: pip install einops).

Jina's code imports a helper that transformers 5 removed, so it is shimmed back in (copied from transformers 4.x).
"""
import sys
from pathlib import Path

import torch
import transformers.models.xlm_roberta.modeling_xlm_roberta as hf_xlmr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convert import MODEL_NAME, REVISION, load_native_model  # noqa: E402


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask + padding_idx


hf_xlmr.create_position_ids_from_input_ids = create_position_ids_from_input_ids
ref = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=REVISION).float().eval()
nat = load_native_model()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
q = ["what is panda?"] * 4 + ["Wie alt ist der Eiffelturm?"] * 2
d = ["hi", "The giant panda is a bear species endemic to China.", "Paris is in France", "Le panda géant est un ours. " * 40,
     "Der Eiffelturm wurde 1889 fertiggestellt.", "Berlin ist die Hauptstadt."]
e = tok(q, d, padding=True, truncation=True, max_length=1024, return_tensors="pt")
with torch.inference_mode():
    a, b = ref(**e).logits.view(-1), nat(**e).logits.view(-1)
diff = (a - b).abs().max().item()
print("jina remote code:", [round(x, 5) for x in a.tolist()])
print("converted       :", [round(x, 5) for x in b.tolist()])
print(f"max |diff| = {diff:.2e}", "PASS" if diff < 1e-4 else "FAIL")
