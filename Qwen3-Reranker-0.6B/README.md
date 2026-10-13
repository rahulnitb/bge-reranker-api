# Qwen3-Reranker-0.6B reranker API (CPU, ONNX)

A FastAPI service that reranks documents against a query using
[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), served on **port 8904**.
License: Apache-2.0.

Qwen3-Reranker is a 0.6B-parameter **decoder LLM** (28 layers, hidden size 1024, GQA 16/8 heads), not a
classifier. Each (query, document) pair is put into a fixed chat prompt that asks the model to answer "yes" or "no";
the score is `sigmoid(logit("yes") - logit("no"))` at the last token, which is the same number as the softmax
probability of "yes" in the official example. It supports a task **instruction** and long inputs, but it is about as
heavy as bge-reranker-v2-m3, so expect seconds per request on CPU.

This folder is self-contained (own venv, model files, logs and scripts) and has the same API as the other
services in this repo, plus one optional `instruction` field.

## Quick start (Linux)

Requirements: Python 3.10+ with the `venv` module, x86_64 CPU, ~8 GB free disk during setup (~4 GB after),
~4 GB RAM, internet access for the first setup.

```bash
cd Qwen3-Reranker-0.6B
./setup.sh     # creates .venv, installs packages, downloads the model, exports + quantizes to ONNX (~5-10 min)
./run.sh       # starts the API on http://0.0.0.0:8904 with the ONNX fp32 backend (interactive docs: /docs)
```

Windows: `.\setup.ps1` and `.\run.ps1`.

## API

Same as the other services in this repo:

- `POST /rerank` with `query`, `documents` (strings or objects such as `{"program_id": 1, "desc": "..."}`),
  `text_field` (default `desc`), `top_n`, `normalize` (sigmoid, default `true`) and `max_length`.
- **`instruction`** (optional, Qwen only): the task description inserted in the prompt. Default:
  `"Given a web search query, retrieve relevant passages that answer the query"`. A task-specific instruction
  (e.g. `"Given a product search query, retrieve matching program descriptions"`) usually helps.
- `max_length` counts the **whole prompt** (~48 tokens of template + instruction + query + document). Default 512,
  allowed 128-8192. When a pair is too long, only the end of the document is cut; the prompt and query stay intact.
- `normalize: false` returns the raw `logit(yes) - logit(no)`.
- `GET /health` shows the status, backend, model file and tokenizer.

```bash
curl -s http://localhost:8904/rerank -H 'content-type: application/json' -d '{
  "query": "what is panda?",
  "documents": [{"program_id": 1, "desc": "hi"}, {"program_id": 2, "desc": "The giant panda is a bear species endemic to China."}],
  "instruction": "Given a question, retrieve passages that answer it",
  "top_n": 1
}'
```

Scores from this model are typically lower and more spread out than from the cross-encoders (e.g. 0.17 for
"what is panda?" vs a correct one-line answer), so compare rankings, not absolute values, across models.

## How it is served

`qwen_scoring.py` holds everything model-specific and is shared by the API, the export and the validation:

- **Prompt building** (`build_inputs`): the template from the model card, left padding so the last position is
  always the real last token, and truncation that only cuts the document.
- **Scoring head** (`YesNoScorer`): runs the transformer body and multiplies the last hidden state with
  `W[yes] - W[no]` from the (tied) LM head. That replaces the 151k-row LM head with a single dot product.
  It matches the official full-LM-head method within 2e-6, including batched and left-padded input.
- The model files come from a pinned revision.

ONNX pipeline:

| Step | Script | Output |
|---|---|---|
| Export the scorer (eager attention, no KV cache) | `onnx/export.py` | `model_fp32.onnx` + `.data` (~2.4 GB) |
| Weight-only int8 (MatMulNBits, block 32) | `onnx/quantize.py` | `model_int8.onnx` + `.data` (~1.1 GB) |
| Save tokenizer for offline use | `onnx/save_tokenizer.py` | `tokenizer/` |

ONNX fp32 matches torch within ~1e-5 on batch 1, unpadded, mixed-length/multilingual and truncated inputs.

### Why there is no fused graph and why int8 is weight-only

- **Fusion:** the ONNX Runtime transformer optimizer (used by the other services) crashes on this >2 GB graph, and
  its attention patterns target BERT/GPT-2, not Qwen3's GQA + RoPE + q/k-norm attention. ONNX Runtime still applies
  its built-in graph optimizations (incl. RMSNorm fusion) when it loads the model.
- **Int8:** measured on `onnx/validate.py` (Spearman vs torch, top-1 = relevant document ranked first, out of 10)
  and on 10 documents x 206 tokens:

| Quantization | Spearman | Top-1 | Max score diff | Time vs fp32 (8.6 s) |
|---|---|---|---|---|
| Dynamic int8, all ops | 0.03 | 2/10 | 0.95 | 7.4 s |
| Dynamic int8, weight MatMuls only | -0.05 | 4/10 | 0.93 | 7.1 s |
| ... and down_proj / o_proj kept fp32 | -0.04 / -0.01 | 1/10 | 0.98 | 7.9 / 8.5 s |
| Weight-only int4, block 32 | 0.92-0.93 | 10/10 | 0.17-0.20 | 12.0-14.5 s |
| **Weight-only int8, block 32 (used for `onnx-int8`)** | **0.995** | **10/10** | **0.005** | 19.9 s |

Standard dynamic int8 quantizes each activation tensor with one scale; this LLM's activations have outliers after
RMSNorm, so that destroys the scores. Weight-only quantization is accurate, but on this AVX2 CPU without VNNI its
kernels are slower than fp32. That is why `run.sh` defaults to the **fp32 ONNX** backend. On a CPU with
AVX512-VNNI/AMX, benchmark `onnx-int8` - it may win there. Faster int8 for this model would need calibrated static
or SmoothQuant-style quantization, which is not done here.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RERANK_BACKEND` | `onnx` via `run.sh` (`torch` if running `app.py` directly) | `torch`, `onnx` (fp32) or `onnx-int8` (weight-only int8) |
| `RERANK_PORT` | `8904` | Port |
| `RERANK_THREADS` | torch: auto, ONNX: all logical cores | CPU threads |
| `RERANK_BATCH_SIZE` | `16` | Documents scored per mini-batch |
| `RERANK_MAX_LENGTH` | `512` | Default max tokens per pair, prompt included (requests may go up to 8192) |
| `RERANK_LOG_FILE` | `logs/requests.jsonl` | Where each request/response is logged |

## Accuracy

`onnx/validate.py` (10 handwritten queries x 4 documents, compared with PyTorch):

| Backend | Spearman vs torch | Top-1 agreement | Max normalized-score diff |
|---|---|---|---|
| ONNX fp32 | 1.0000 | 100% | 0.0000 |
| ONNX int8 (weight-only) | 0.995 | 100% | 0.005 |

All backends put the relevant document first for 10/10 queries. This is a small sanity check, not a full
quality evaluation.

## Measured latency

Same machine as the rest of the repo (Windows 11, Intel i5-8250U: 4 cores / 8 threads, AVX2, no VNNI). p50 latency
through the API, only 3 measured requests per cell after warmup (`bench/benchmark.py --quick --requests 3`), so
indicative only. Short = ~25 words per document, medium = ~150 words; every pair also carries the ~48-token prompt.

| Docs | Length | Concurrency | torch | ONNX fp32 (default) | ONNX int8 (weight-only) | fp32 vs torch |
|---|---|---|---|---|---|---|
| 1 | short | 1 | 597 ms | 467 ms | 2.54 s | 1.3x |
| 10 | short | 1 | 6.18 s | 5.07 s | 10.18 s | 1.2x |
| 50 | short | 1 | 33.80 s | 23.27 s | 45.74 s | 1.5x |
| 1 | medium | 1 | 1.77 s | 956 ms | 2.55 s | 1.9x |
| 10 | medium | 1 | 14.54 s | 10.97 s | 20.48 s | 1.3x |
| 50 | medium | 1 | 73.58 s | 51.46 s | 98.69 s | 1.4x |
| 10 | medium | 4 | 38.68 s | 39.30 s | 53.53 s | 1.0x |

On this CPU Qwen3-Reranker is the slowest model in the repo: 10 medium documents take 11 s (ONNX fp32) vs 6.1 s
for bge-reranker-v2-m3 int8, 1.6 s for Jina int8 and 0.40 s for MiniLM int8. Use it when its ranking quality or
instruction support matters more than latency, or on faster hardware.
Concurrency does not add throughput: requests share the same CPU. Raw results: `bench/results_<backend>.md`.

## Files

| Path | Purpose |
|---|---|
| `app.py` | The API |
| `qwen_scoring.py` | Prompt template, document-only truncation, yes/no scoring head (pinned revision) |
| `run.sh` / `setup.sh` | Start the server / one-time setup (Linux). `run.ps1` / `setup.ps1` for Windows |
| `onnx/export.py` | Scorer -> ONNX fp32 (external data) |
| `onnx/quantize.py` | fp32 -> weight-only int8 |
| `onnx/save_tokenizer.py` | Saves the tokenizer next to the ONNX files |
| `onnx/validate.py` | Accuracy check: PyTorch vs ONNX fp32 vs int8 |
| `bench/benchmark.py` | Latency benchmark against a running server (`--url`, default port 8904) |
