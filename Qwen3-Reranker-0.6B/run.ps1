# Start the reranker API on port 8904 using this folder's venv.
# Backend: torch | onnx | onnx-int8 (default here: onnx - fp32 is the fastest accurate option on AVX2 CPUs, see README)
if (-not $env:RERANK_BACKEND) { $env:RERANK_BACKEND = "onnx" }
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
