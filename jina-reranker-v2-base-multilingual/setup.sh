#!/usr/bin/env bash
# One-command setup (Linux/macOS): venv -> dependencies -> download model -> ONNX export -> graph fusion -> int8 quantize -> save tokenizer.
# Needs Python 3.10+ (with the venv module) and ~4 GB free disk.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
[ -d .venv ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python

$PY -m pip install --upgrade pip
$PY -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -r requirements.txt

if [ ! -f onnx/models/model_int8.onnx ]; then
    $PY onnx/export.py            # downloads jinaai/jina-reranker-v2-base-multilingual (~560 MB), maps it to standard XLM-R -> model_fp32.onnx
    $PY onnx/optimize.py          # fused Attention/SkipLayerNorm/BiasGelu -> onnx/models/model_opt.onnx
    $PY onnx/quantize.py          # -> onnx/models/model_int8.onnx (feed-forward MatMuls only)
    $PY onnx/save_tokenizer.py    # -> onnx/models/tokenizer/
else
    echo "ONNX models already present, skipping build."
fi
echo
echo "Done. Start the API with:  ./run.sh"
