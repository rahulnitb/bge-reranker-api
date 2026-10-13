"""Save the tokenizer next to the ONNX files so the ONNX backends run without Hugging Face."""
from pathlib import Path

import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convert import MODEL_NAME, REVISION  # noqa: E402

out = Path(__file__).parent / "models" / "tokenizer"
AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION).save_pretrained(out)
print("saved", out)
