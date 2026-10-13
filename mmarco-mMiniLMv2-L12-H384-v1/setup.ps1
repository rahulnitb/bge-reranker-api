# One-command setup: venv -> dependencies -> download model -> ONNX export -> graph fusion -> int8 quantize -> save tokenizer.
# Needs Python 3.10+ and ~2 GB free disk.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) { python -m venv .venv }
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt

if (-not (Test-Path "onnx\models\model_int8.onnx")) {
    & $py onnx\export.py            # downloads cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~470 MB) -> onnx\models\model_fp32.onnx
    & $py onnx\optimize.py          # fused Attention/SkipLayerNorm/BiasGelu -> onnx\models\model_opt.onnx
    & $py onnx\quantize.py          # -> onnx\models\model_int8.onnx
    & $py onnx\save_tokenizer.py    # -> onnx\models\tokenizer\
} else {
    Write-Host "ONNX models already present, skipping build."
}
Write-Host "`nDone. Start the API with:  .\run.ps1"
