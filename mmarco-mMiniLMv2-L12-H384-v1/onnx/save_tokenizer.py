"""Save the tokenizer next to the ONNX files so the ONNX backends run without Hugging Face."""
from pathlib import Path

from transformers import AutoTokenizer

out = Path(__file__).parent / "models" / "tokenizer"
AutoTokenizer.from_pretrained("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").save_pretrained(out)
print("saved", out)
