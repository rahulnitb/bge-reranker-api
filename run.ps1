# Start the reranker API on port 8901 using the project venv.
# Backend: torch | onnx | onnx-int8 (default here: onnx-int8)
if (-not $env:RERANK_BACKEND) { $env:RERANK_BACKEND = "onnx-int8" }
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
