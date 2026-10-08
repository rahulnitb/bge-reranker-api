# bge-reranker-v2-m3 API (CPU, ONNX int8)

A FastAPI service that reranks documents against a query using
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), served on **port 8901**.
It runs on CPU and supports three backends: PyTorch, ONNX fp32 and ONNX int8 (default, fastest).

## Quick start (Linux)

Requirements: Python 3.10+ with the `venv` module (on Debian/Ubuntu: `sudo apt install python3-venv`),
`git`, x86_64 CPU, ~8 GB free disk, internet access for the first setup.

```bash
git clone https://github.com/rahulnitb/bge-reranker-api.git
cd bge-reranker-api
./setup.sh     # creates .venv, installs packages, downloads the model, exports + quantizes to ONNX (~5-10 min)
./run.sh       # starts the API on http://0.0.0.0:8901  (interactive docs: /docs)
```

If `./setup.sh` is not executable: `chmod +x setup.sh run.sh`, or run `bash setup.sh`.
To use a specific interpreter: `PYTHON=python3.12 ./setup.sh`.

The model files (~3 GB) are **not** stored in this repo. `setup.sh` rebuilds them with
`onnx/export.py` -> `onnx/quantize.py` -> `onnx/save_tokenizer.py`. After setup, the ONNX backends
run fully offline from `onnx/models/`.

Run in the background / keep it running after logout:
```bash
nohup ./run.sh > server.log 2>&1 &          # simple
# or as a systemd service: ExecStart=/path/to/bge-reranker-api/run.sh
```
Stop it with `pkill -f "python app.py"` (or `kill <pid>`). Open port 8901 in your firewall if calling it from another machine.

Windows? Use `setup.ps1` / `run.ps1` instead (same steps).

## API

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
| `documents` | required | List of strings **or objects** such as `{"program_id": 1, "desc": "..."}` (at least 1, no upper limit) |
| `text_field` | `desc` | For object documents: the key that holds the text to rank |
| `top_n` | all | Return only the best N |
| `normalize` | `true` | Sigmoid scores in 0..1; `false` gives raw logits |
| `max_length` | 512 | Max tokens per query+document pair (longer is truncated) |

Response (sorted best first; `index` is the position in your input):
```json
{
  "results": [{"index": 1, "score": 0.8638, "document": "The giant panda is ..."}, {"index": 0, "score": 0.0002, "document": "hi"}],
  "latency_ms": 115.2
}
```
`latency_ms` is server-side scoring time.

### Tracking documents by ID
Send objects instead of strings. The text is read from the `desc` key (change it with `text_field`); all other
keys (`program_id`, `category`, ...) are passed through untouched, and each result echoes the whole object in `document`:
```bash
curl -s http://localhost:8901/rerank -H 'content-type: application/json' -d '{
  "query": "what is panda?",
  "documents": [
    {"program_id": 1, "desc": "hi"},
    {"program_id": 2, "desc": "The giant panda is a bear species endemic to China."}
  ]
}'
# -> {"results":[{"index":1,"score":0.86,"document":{"program_id":2,"desc":"The giant panda ..."}},
#                {"index":0,"score":0.0002,"document":{"program_id":1,"desc":"hi"}}],"latency_ms":...}
```
Read the ID with `result["document"]["program_id"]`. An object without a string `desc` field returns HTTP 422.
Plain strings still work, and strings and objects can be mixed in one request.

### `GET /health`
Shows status, backend, model file and tokenizer location in use.

### Examples
```bash
# curl
curl -s http://localhost:8901/rerank -H 'content-type: application/json' -d '{
  "query": "what is panda?",
  "documents": ["hi", "The giant panda is a bear species endemic to China.", "Paris is in France"],
  "top_n": 2
}'

curl -s http://localhost:8901/health
```
```python
# Python (run with .venv/bin/python, or any environment that has httpx)
import httpx
r = httpx.post("http://localhost:8901/rerank", json={
    "query": "what is panda?",
    "documents": ["hi", "The giant panda is a bear species endemic to China."],
}, timeout=300)
print(r.json())
```
Set a long client timeout: many long documents take tens of seconds on CPU.

## Configuration (environment variables)

Set them in front of the command, e.g. `RERANK_BACKEND=torch RERANK_THREADS=4 ./run.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `RERANK_BACKEND` | `onnx-int8` via `run.sh` (`torch` if running `app.py` directly) | `torch`, `onnx` (fp32) or `onnx-int8` |
| `RERANK_PORT` | `8901` | Port |
| `RERANK_THREADS` | torch: auto, ONNX: all logical cores | CPU threads |
| `RERANK_BATCH_SIZE` | `16` | Documents scored per mini-batch |
| `RERANK_MAX_LENGTH` | `512` | Default max tokens |
| `RERANK_LOG_FILE` | `logs/requests.jsonl` | Where each request/response is logged |

Every call is appended to `logs/requests.jsonl` as one JSON line: timestamp, number of documents, latency, full request and full response.
View the latest entry: `tail -n 1 logs/requests.jsonl | python3 -m json.tool`.

## Files

| Path | Purpose |
|---|---|
| `app.py` | The API |
| `run.sh` / `setup.sh` | Start the server / one-time setup (Linux). `run.ps1` / `setup.ps1` are the Windows versions |
| `onnx/export.py` | PyTorch -> ONNX fp32 |
| `onnx/quantize.py` | fp32 -> dynamic int8 |
| `onnx/save_tokenizer.py` | Saves the tokenizer next to the ONNX files |
| `onnx/validate.py` | Accuracy check: PyTorch vs ONNX fp32 vs int8 |
| `bench/benchmark.py` | Latency benchmark against a running server |
| `bench/compare.py` | Starts each backend and compares them |

## Latency test plan

With the server running (in another terminal):
```bash
.venv/bin/python bench/benchmark.py --quick   # smaller grid
.venv/bin/python bench/benchmark.py           # full grid (slow on CPU)
.venv/bin/python bench/compare.py             # torch vs onnx vs onnx-int8, starts its own servers on port 8902
.venv/bin/python onnx/validate.py             # accuracy: torch vs onnx vs int8
```
`benchmark.py` measures, for each scenario, after warmup requests: p50/p95/p99/mean/min/max round-trip latency,
server-side time, requests/s, documents/s and error rate. Scenarios: documents per request (1, 10, 50, 100) x
document length (short ~25 words, medium ~150, long ~450) at concurrency 1, plus concurrency 4 and 8 at
10 medium documents. Results are written to `bench/results.csv` and `bench/results.md`. The data is synthetic;
edit `make_payload` in `bench/benchmark.py` to use real text.

## Measured results

These numbers come from the original development machine (Windows 11, Intel i5-8250U: 4 cores / 8 threads,
AVX2, no VNNI, 16 GB RAM), not from Linux, so re-run the benchmark on your hardware. One request at a time,
medium-length (~150 word) documents, p50 latency:

| Documents | torch | ONNX fp32 | ONNX int8 | int8 speedup |
|---|---|---|---|---|
| 1 | 1.11 s | 0.82 s | 0.95 s | 1.17x |
| 10 | 10.5 s | 9.1 s | 6.1 s | 1.70x |
| 50 | 54.3 s | 48.0 s | 33.8 s | 1.61x |

(3 measured requests per cell, so indicative only.) 20 documents of ~115-130 words took 8.7 s on the int8
backend. Concurrency does not add throughput: requests share the same CPU.

Accuracy (`onnx/validate.py`, 10 handwritten queries x 4 documents): ONNX fp32 matches PyTorch exactly;
int8 has Spearman 0.991 against PyTorch with identical top-1 results (max normalized-score difference 0.027).
This is a small sanity check, not a full quality evaluation.

## Troubleshooting

- `ensurepip is not available` / venv errors: install `python3-venv` (Debian/Ubuntu) or the equivalent package.
- `Tokenizer not found ... Run: python onnx/save_tokenizer.py`: setup did not finish; re-run `./setup.sh`.
- Out-of-memory during export: it needs roughly 6-8 GB RAM free; close other programs or add swap.
- `Permission denied: ./setup.sh`: run `chmod +x setup.sh run.sh`.
- Using int8 on a CPU with AVX512-VNNI/AMX gives a bigger speedup than the numbers above.

## Notes

- CPU only. A GPU or newer CPU will be much faster.
- To speed things up further: lower `max_length`, send fewer candidate documents, or use a smaller reranker.
- The model is released by BAAI under its own license; see the model card.
