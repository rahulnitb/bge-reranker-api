import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer

from convert import MODEL_NAME, REVISION, load_native_model

BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
MODEL_MAX_LENGTH = 1024  # model has 1026 position embeddings (XLM-R style), so 1024 tokens max
DEFAULT_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
BACKEND = os.getenv("RERANK_BACKEND", "torch")  # torch | onnx | onnx-opt | onnx-int8
PORT = int(os.getenv("RERANK_PORT", "8903"))
ONNX_DIR = Path(__file__).parent / "onnx" / "models"
ONNX_FILES = {"onnx": "model_fp32.onnx", "onnx-opt": "model_opt.onnx", "onnx-int8": "model_int8.onnx"}

LOG_FILE = Path(os.getenv("RERANK_LOG_FILE", Path(__file__).parent / "logs" / "requests.jsonl"))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

state = {}


def log_call(request: dict, response: dict, latency_ms: float):
    """Append one JSON line per call: timestamp, full request, full response, timing."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_documents": len(request["documents"]),
        "latency_ms": round(latency_ms, 1),
        "request": request,
        "response": response,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threads = os.getenv("RERANK_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    t0 = time.perf_counter()
    if BACKEND == "torch":
        state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
        state["model"] = load_native_model()  # Jina weights in the built-in XLM-R class, see convert.py
        state["model_source"] = MODEL_NAME
        state["tokenizer_source"] = MODEL_NAME
    else:
        import onnxruntime as ort

        # ONNX backends run fully from onnx/models/ (no Hugging Face access needed)
        tok_dir = ONNX_DIR / "tokenizer"
        if not tok_dir.exists():
            raise RuntimeError(f"Tokenizer not found at {tok_dir}. Run: python onnx/save_tokenizer.py")
        state["tokenizer"] = AutoTokenizer.from_pretrained(str(tok_dir))
        state["tokenizer_source"] = str(tok_dir)

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads) if threads else (os.cpu_count() or 4)
        if BACKEND not in ONNX_FILES:
            raise RuntimeError(f"Unknown RERANK_BACKEND={BACKEND!r}; use torch | {' | '.join(ONNX_FILES)}")
        fname = ONNX_FILES[BACKEND]
        state["model"] = ort.InferenceSession(str(ONNX_DIR / fname), so, providers=["CPUExecutionProvider"])
        state["model_source"] = str(ONNX_DIR / fname)
    score_pairs("warmup query", ["warmup document"], DEFAULT_MAX_LENGTH, True)
    state["load_s"] = time.perf_counter() - t0
    yield


app = FastAPI(title="jina-reranker-v2-base-multilingual", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    # each document is a plain string, or an object like {"program_id": 1, "desc": "..."}
    documents: List[Union[str, Dict[str, Any]]] = Field(min_length=1)
    text_field: str = "desc"  # which key of an object document holds the text to rank
    top_n: Optional[int] = Field(default=None, ge=1)
    normalize: bool = True
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=MODEL_MAX_LENGTH)

    @model_validator(mode="after")
    def _check_text_field(self):
        for i, d in enumerate(self.documents):
            if isinstance(d, dict) and not isinstance(d.get(self.text_field), str):
                raise ValueError(f"documents[{i}] must have a string field '{self.text_field}'")
        return self

    def texts(self) -> List[str]:
        return [d if isinstance(d, str) else d[self.text_field] for d in self.documents]


class RerankItem(BaseModel):
    index: int
    score: float
    document: Union[str, Dict[str, Any]]  # echoes the input document unchanged (string or object)


class RerankResponse(BaseModel):
    results: List[RerankItem]
    latency_ms: float


def score_pairs(query, docs, max_length, normalize):
    tok, model = state["tokenizer"], state["model"]
    # sort by length so each mini-batch pads to similar lengths
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    scores = [0.0] * len(docs)
    with torch.inference_mode():
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE]
            chunk = [docs[j] for j in idx]
            if BACKEND == "torch":
                inputs = tok([query] * len(chunk), chunk, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt")
                logits = model(**inputs).logits.view(-1).float()
            else:
                inputs = tok([query] * len(chunk), chunk, padding=True, truncation=True,
                             max_length=max_length, return_tensors="np")
                out = model.run(None, {"input_ids": inputs["input_ids"].astype("int64"),
                                       "attention_mask": inputs["attention_mask"].astype("int64")})[0]
                logits = torch.from_numpy(out).view(-1).float()
            if normalize:
                logits = torch.sigmoid(logits)
            for j, v in zip(idx, logits.tolist()):
                scores[j] = v
    return scores


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):  # sync def -> runs in threadpool, event loop stays free
    t0 = time.perf_counter()
    scores = score_pairs(req.query, req.texts(), req.max_length, req.normalize)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if req.top_n:
        order = order[: req.top_n]
    results = [RerankItem(index=i, score=scores[i], document=req.documents[i]) for i in order]
    latency_ms = (time.perf_counter() - t0) * 1000
    resp = RerankResponse(results=results, latency_ms=latency_ms)
    log_call(req.model_dump(), resp.model_dump(), latency_ms)
    return resp


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in state else "loading",
        "base_model": MODEL_NAME,
        "model_file": state.get("model_source"),
        "tokenizer": state.get("tokenizer_source"),
        "device": "cpu",
        "backend": BACKEND,
        "torch_threads": torch.get_num_threads(),
        "load_seconds": state.get("load_s"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
