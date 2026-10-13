#!/usr/bin/env bash
# Start the reranker API on port 8904 using this folder's venv.
# Backend: torch | onnx | onnx-int8 (default here: onnx - fp32 is the fastest accurate option on AVX2 CPUs, see README)
cd "$(dirname "$0")"
export RERANK_BACKEND="${RERANK_BACKEND:-onnx}"
exec .venv/bin/python app.py
