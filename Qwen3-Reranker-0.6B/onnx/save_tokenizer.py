"""Save the tokenizer next to the ONNX files so the ONNX backends run without Hugging Face."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_scoring import load_tokenizer  # noqa: E402

out = Path(__file__).parent / "models" / "tokenizer"
load_tokenizer().save_pretrained(out)
print("saved", out)
