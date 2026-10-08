"""Save the tokenizer next to the ONNX files so the ONNX backends run without Hugging Face."""
from pathlib import Path

from transformers import AutoTokenizer

out = Path(__file__).parent / "models" / "tokenizer"
AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3").save_pretrained(out)
print("saved", out)
