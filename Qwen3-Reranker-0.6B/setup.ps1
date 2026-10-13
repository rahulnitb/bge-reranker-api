# One-command setup: venv -> dependencies -> download model -> ONNX export -> int8 quantize -> save tokenizer.
# Needs Python 3.10+ and ~8 GB free disk (intermediate files are deleted afterwards).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) { python -m venv .venv }
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt

if (-not (Test-Path "onnx\models\model_int8.onnx")) {
    & $py onnx\export.py            # downloads Qwen/Qwen3-Reranker-0.6B (~1.2 GB) -> model_fp32.onnx (+ .data, ~2.4 GB)
    & $py onnx\quantize.py          # -> onnx/models/model_int8.onnx
    & $py onnx\save_tokenizer.py    # -> onnx\models\tokenizer\
    Remove-Item -Recurse -Force onnx\models\raw, onnx\models\pre -ErrorAction SilentlyContinue
} else {
    Write-Host "ONNX models already present, skipping build."
}
Write-Host "`nDone. Start the API with:  .\run.ps1"
