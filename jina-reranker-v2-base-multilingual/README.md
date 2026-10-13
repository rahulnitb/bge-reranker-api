# jina-reranker-v2-base-multilingual reranker API (CPU, ONNX fused + int8)

A FastAPI service that reranks documents against a query using
[jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual),
served on **port 8903**. It is a multilingual XLM-R base cross-encoder (12 layers, hidden size 768, ~278M
parameters) that accepts up to **1024 tokens** per query+document pair.

> **License: CC-BY-NC-4.0 (non-commercial use only).** See the model card.

This folder is self-contained (own venv, model files, logs and scripts) and has the same API as the other
services in this repo, so clients only need to change the port.

## Quick start (Linux)

Requirements: Python 3.10+ with the `venv` module, x86_64 CPU, ~4 GB free disk, internet access for the first setup.

```bash
cd jina-reranker-v2-base-multilingual
./setup.sh     # creates .venv, installs packages, downloads + converts the model, exports/fuses/quantizes to ONNX (~3-5 min)
./run.sh       # starts the API on http://0.0.0.0:8903  (interactive docs: /docs)
```

Windows: `.\setup.ps1` and `.\run.ps1`.

## No remote code: how the model is loaded

Jina publishes the model with custom flash-attention code (`trust_remote_code`) written for transformers 4.x.
It does not import on transformers 5. The architecture is a plain XLM-R base (post-norm, absolute positions,
exact GELU); only the weight names and layout differ. `convert.py` maps Jina's weights onto the built-in
`XLMRobertaForSequenceClassification`, splitting the fused `Wqkv` into query/key/value and renaming the rest.
No remote code runs, and the files come from a pinned revision.

`onnx/check_conversion.py` proves the mapping is exact. It loads Jina's original code with a one-line
compatibility shim and compares the two on English, German, French and a long padded document. The max logit
difference is 6e-7. Running it needs `pip install einops`.

## ONNX pipeline

| Step | Script | Output |
|---|---|---|
| Convert + export to ONNX | `onnx/export.py` | `model_fp32.onnx` (~1.1 GB) |
| Fuse transformer ops (Attention, SkipLayerNorm, BiasGelu) | `onnx/optimize.py` | `model_opt.onnx` (~1.1 GB) |
| Dynamic int8, per-channel, **feed-forward MatMuls only** | `onnx/quantize.py` | `model_int8.onnx` (~920 MB) |
| Save tokenizer for offline use | `onnx/save_tokenizer.py` | `tokenizer/` |

Export and fusion use the same graph rewrites as the MiniLM service (see its README). All 12 attention blocks fuse.

**Why int8 leaves attention and embeddings in fp32:** quantizing every op, as the MiniLM service does, dropped
Spearman against torch to 0.83 for this model. The fused Attention op's QKV projection is the sensitive part. Measured
int8 variants (10 handwritten queries x 4 docs; timings: 10 docs of ~170 tokens, ONNX Runtime directly):

| int8 variant | Spearman vs torch | Max score diff | Size | Time |
|---|---|---|---|---|
| fp32 fused (reference) | 1.000 | 0.000 | 1114 MB | 1.32 s |
| all ops (MiniLM recipe) | 0.835 | 0.199 | 279 MB | ~same as below |
| **feed-forward MatMuls only (used)** | **0.987** | **0.032** | 921 MB | 0.93 s |
| MatMuls + embedding Gather | 0.978 | 0.027 | 343 MB | 0.90 s |
| Jina's published `onnx/model_int8.onnx` | 0.983 | 0.061 | 280 MB | 0.97 s |

All variants rank the relevant document first for 10/10 queries. Fusion alone gives little on this CPU; the
speedup comes from int8. To compare against Jina's published int8 file yourself, put it at
`onnx/models/jina_official/model_int8.onnx` and `onnx/validate.py` includes it automatically.

## API

Identical to the other services in this repo:

- `POST /rerank` with `query`, `documents` (strings or objects such as `{"program_id": 1, "desc": "..."}`),
  `text_field` (default `desc`), `top_n`, `normalize` (sigmoid, default `true`) and `max_length`.
  For this model `max_length` defaults to 512 and can be raised to **1024**.
- `GET /health` shows the status, backend, model file and tokenizer.

```bash
curl -s http://localhost:8903/rerank -H 'content-type: application/json' -d '{
  "query": "what is panda?",
  "documents": [{"program_id": 1, "desc": "hi"}, {"program_id": 2, "desc": "The giant panda is a bear species endemic to China."}],
  "top_n": 1
}'
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RERANK_BACKEND` | `onnx-int8` via `run.sh` (`torch` if running `app.py` directly) | `torch`, `onnx` (fp32), `onnx-opt` (fused fp32) or `onnx-int8` |
| `RERANK_PORT` | `8903` | Port |
| `RERANK_THREADS` | torch: auto, ONNX: all logical cores | CPU threads |
| `RERANK_BATCH_SIZE` | `16` | Documents scored per mini-batch |
| `RERANK_MAX_LENGTH` | `512` | Default max tokens (requests may go up to 1024) |
| `RERANK_LOG_FILE` | `logs/requests.jsonl` | Where each request/response is logged |

## Accuracy

`onnx/validate.py` (10 handwritten queries x 4 documents, compared with PyTorch):

| Backend | Spearman vs torch | Top-1 agreement | Max normalized-score diff |
|---|---|---|---|
| ONNX fp32 | 1.0000 | 100% | 0.0000 |
| ONNX fused (`onnx-opt`) | 1.0000 | 100% | 0.0000 |
| ONNX int8 | 0.987 | 100% | 0.032 |

This is a small sanity check, not a full quality evaluation. Individual int8 scores can move more than the table
suggests: in the API smoke test, "what is panda?" vs "The giant panda is a bear species endemic to China." scored
0.68 on torch and 0.76 on int8, with the same ranking. If you threshold on absolute scores, use `onnx-opt`.

## Measured latency

Same machine as the rest of the repo (Windows 11, Intel i5-8250U: 4 cores / 8 threads, AVX2, no VNNI). p50 latency
through the API, 10 measured requests per cell after warmup (`bench/benchmark.py --quick --requests 10`), so
indicative only; single-document rows are noisy. Short = ~25 words per document, medium = ~150 words.

| Docs | Length | Concurrency | torch | ONNX fp32 | ONNX fused | ONNX int8 | int8 vs torch |
|---|---|---|---|---|---|---|---|
| 1 | short | 1 | 93 ms | 51 ms | 66 ms | 84 ms | 1.1x |
| 10 | short | 1 | 590 ms | 378 ms | 406 ms | 311 ms | 1.9x |
| 50 | short | 1 | 3.39 s | 2.36 s | 2.38 s | 1.83 s | 1.9x |
| 1 | medium | 1 | 279 ms | 155 ms | 195 ms | 172 ms | 1.6x |
| 10 | medium | 1 | 2.31 s | 2.02 s | 2.02 s | 1.61 s | 1.4x |
| 50 | medium | 1 | 11.92 s | 10.23 s | 9.82 s | 8.41 s | 1.4x |
| 10 | medium | 4 | 9.38 s | 7.72 s | 7.58 s | 6.10 s | 1.5x |

Compared with the other services (int8, medium documents): bge-reranker-v2-m3 takes 6.1 s for 10 / 33.8 s for 50,
this model 1.6 s / 8.4 s (~4x faster than bge), MiniLM 0.40 s / 2.95 s (~4x faster than this).
Concurrency does not add throughput: requests share the same CPU. Raw results: `bench/results_<backend>.md`.

## Files

| Path | Purpose |
|---|---|
| `app.py` | The API |
| `convert.py` | Loads Jina's weights into the built-in XLM-R class (pinned revision) |
| `run.sh` / `setup.sh` | Start the server / one-time setup (Linux). `run.ps1` / `setup.ps1` for Windows |
| `onnx/export.py` | PyTorch -> ONNX fp32 (fusion-friendly graph) |
| `onnx/optimize.py` | Graph rewrites + ONNX Runtime transformer fusion |
| `onnx/quantize.py` | Fused graph -> int8 (feed-forward MatMuls) |
| `onnx/save_tokenizer.py` | Saves the tokenizer next to the ONNX files |
| `onnx/validate.py` | Accuracy check: PyTorch vs ONNX fp32 vs fused vs int8 |
| `onnx/check_conversion.py` | Optional: converted model vs Jina's original remote code |
| `bench/benchmark.py` | Latency benchmark against a running server (`--url`, default port 8903) |
