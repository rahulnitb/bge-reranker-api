#!/usr/bin/env bash
# Start the reranker API on port 8902 using this folder's venv.
# Backend: torch | onnx | onnx-opt | onnx-int8 (default here: onnx-int8)
cd "$(dirname "$0")"
export RERANK_BACKEND="${RERANK_BACKEND:-onnx-int8}"
exec .venv/bin/python app.py
