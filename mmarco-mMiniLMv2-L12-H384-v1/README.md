# mmarco-mMiniLMv2-L12-H384-v1 reranker API (CPU, ONNX fused + int8)

A FastAPI service that reranks documents against a query using
[cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1),
served on **port 8902**. It is a small multilingual cross-encoder (12 layers, hidden size 384, ~118M parameters,
XLM-R tokenizer, trained on mMARCO), much faster on CPU than bge-reranker-v2-m3 in the parent folder,
at some cost in ranking quality.

This folder is self-contained: its own venv, model files, logs and scripts. The API is the same as the
bge service in the repo root, so clients only need to change the port.

## Quick start (Linux)

Requirements: Python 3.10+ with the `venv` module, x86_64 CPU, ~2 GB free disk, internet access for the first setup.

```bash
cd mmarco-mMiniLMv2-L12-H384-v1
./setup.sh     # creates .venv, installs packages, downloads the model, exports + fuses + quantizes to ONNX (~2-4 min)
./run.sh       # starts the API on http://0.0.0.0:8902  (interactive docs: /docs)
```

Windows: `.\setup.ps1` and `.\run.ps1` (same steps).

The model files are **not** stored in git. Setup rebuilds them into `onnx/models/`:

| Step | Script | Output |
|---|---|---|
| Export PyTorch -> ONNX | `onnx/export.py` | `model_fp32.onnx` (~470 MB) |
| Fuse transformer ops (Attention, SkipLayerNorm, BiasGelu) | `onnx/optimize.py` | `model_opt.onnx` (~470 MB) |
| Dynamic int8 quantization (per-channel) of the fused graph | `onnx/quantize.py` | `model_int8.onnx` (~118 MB) |
| Save tokenizer for offline use | `onnx/save_tokenizer.py` | `tokenizer/` |

After setup, the ONNX backends run fully offline from `onnx/models/`.

Run in the background: `nohup ./run.sh > server.log 2>&1 &`. Stop it with `pkill -f "python app.py"`.

## API

Identical to the root bge service (see the root README for more examples).

### `POST /rerank`
```json
{
  "query": "what is panda?",
  "documents": ["hi", "The giant panda is a bear species endemic to China.", "Paris is in France"],
  "top_n": 2,
  "normalize": true,
  "max_length": 512
}
```
| Field | Default | Meaning |
|---|---|---|
| `query` | required | Search text |
| `documents` | required | List of strings **or objects** such as `{"program_id": 1, "desc": "..."}` |
| `text_field` | `desc` | For object documents: the key that holds the text to rank |
| `top_n` | all | Return only the best N |
| `normalize` | `true` | Sigmoid scores in 0..1; `false` gives raw logits |
| `max_length` | 512 | Max tokens per query+document pair, 16-512 (the model's limit is 512) |

Response (sorted best first; `index` is the position in your input; `document` echoes the input string or object):
```json
{"results": [{"index": 1, "score": 0.97, "document": "The giant panda is ..."}, ...], "latency_ms": 12.3}
```

### `GET /health`
Shows status, backend, model file and tokenizer location in use.

```bash
curl -s http://localhost:8902/rerank -H 'content-type: application/json' -d '{
  "query": "what is panda?",
  "documents": [{"program_id": 1, "desc": "hi"}, {"program_id": 2, "desc": "The giant panda is a bear species endemic to China."}]
}'
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RERANK_BACKEND` | `onnx-int8` via `run.sh` (`torch` if running `app.py` directly) | `torch`, `onnx` (fp32), `onnx-opt` (fused fp32) or `onnx-int8` (fused + int8) |
| `RERANK_PORT` | `8902` | Port |
| `RERANK_THREADS` | torch: auto, ONNX: all logical cores | CPU threads |
| `RERANK_BATCH_SIZE` | `16` | Documents scored per mini-batch |
| `RERANK_MAX_LENGTH` | `512` | Default max tokens |
| `RERANK_LOG_FILE` | `logs/requests.jsonl` | Where each request/response is logged |

Every call is appended to `logs/requests.jsonl` (timestamp, document count, latency, full request and response).

## How the ONNX optimization works

`onnx/optimize.py` runs the ONNX Runtime transformer optimizer, which replaces each layer's ~40 small ops with
fused `Attention`, `SkipLayerNormalization` and `BiasGelu` kernels. Out of the box it fuses nothing for
attention with transformers 5 exports, so two exact graph rewrites happen first:

1. `export.py` exports through a small wrapper that builds the classic additive mask `(1 - mask) * -10000`
   (transformers 5 builds the mask with Gather/And/Where ops the optimizer does not recognize). The export
   asserts the wrapper gives the same logits as the Hugging Face model.
2. `optimize.py` rewrites the score scaling `Mul(QK, 1/sqrt(d))` as `Div(QK, sqrt(d))`, and pins the `-1` in the
   Q/K/V reshapes to the real head count (12). Both are numerically identical.

Result: 12 `Attention`, 24 `SkipLayerNormalization`, 12 `BiasGelu` fused ops. Int8 quantization is then applied to
the fused graph, so both gains stack.

## Accuracy

`onnx/validate.py` (10 handwritten queries x 4 documents, compared with PyTorch):

| Backend | Spearman vs torch | Top-1 agreement | Max normalized-score diff |
|---|---|---|---|
| ONNX fp32 | 1.0000 | 100% | 0.0000 |
| ONNX fused (`onnx-opt`) | 1.0000 | 100% | 0.0000 |
| ONNX int8 | 0.984 | 100% | 0.151 |

All backends put the relevant document first for 10/10 queries. Int8 passes the bar (Spearman >= 0.98, top-1
>= 95%) but moves individual scores more than it does on bge-m3. If you threshold on absolute scores, prefer
`onnx-opt`. This is a small sanity check, not a full quality evaluation.

## Measured latency

Same machine as the root project (Windows 11, Intel i5-8250U: 4 cores / 8 threads, AVX2, no VNNI). p50 latency
through the API, 10 measured requests per cell after warmup (`bench/benchmark.py --quick --requests 10`), so
indicative only. Short = ~25 words per document, medium = ~150 words.

| Docs | Length | Concurrency | torch | ONNX fp32 | ONNX fused | ONNX int8 | int8 vs torch |
|---|---|---|---|---|---|---|---|
| 1 | short | 1 | 35 ms | 18 ms | 16 ms | 13 ms | 2.7x |
| 10 | short | 1 | 194 ms | 116 ms | 110 ms | 94 ms | 2.1x |
| 50 | short | 1 | 883 ms | 591 ms | 597 ms | 872 ms | 1.0x |
| 1 | medium | 1 | 93 ms | 54 ms | 58 ms | 40 ms | 2.3x |
| 10 | medium | 1 | 820 ms | 578 ms | 497 ms | 400 ms | 2.1x |
| 50 | medium | 1 | 4.42 s | 3.33 s | 3.24 s | 2.95 s | 1.5x |
| 10 | medium | 4 | 2.64 s | 2.72 s | 2.29 s | 1.82 s | 1.4x |

Compared with bge-reranker-v2-m3 int8 on the same machine (root README): 10 medium documents 6.1 s -> 0.40 s
(~15x faster), 50 medium documents 33.8 s -> 2.95 s (~11x faster).

The 50-short-document int8 result was slow in two separate runs; fused fp32 (`onnx-opt`) is faster there.
Concurrency does not add throughput: requests share the same CPU. Raw results: `bench/results_<backend>.md`.

## Files

| Path | Purpose |
|---|---|
| `app.py` | The API |
| `run.sh` / `setup.sh` | Start the server / one-time setup (Linux). `run.ps1` / `setup.ps1` for Windows |
| `onnx/export.py` | PyTorch -> ONNX fp32 (fusion-friendly graph) |
| `onnx/optimize.py` | Graph rewrites + ONNX Runtime transformer fusion |
| `onnx/quantize.py` | Fused graph -> dynamic int8 |
| `onnx/save_tokenizer.py` | Saves the tokenizer next to the ONNX files |
| `onnx/validate.py` | Accuracy check: PyTorch vs ONNX fp32 vs fused vs int8 |
| `bench/benchmark.py` | Latency benchmark against a running server (`--url`, default port 8902) |

## Notes

- The model is released by its authors under the license on the model card.
- CPU only. A CPU with AVX512-VNNI/AMX gets a bigger int8 speedup.
