"""Weight-only int8 quantization (MatMulNBits, block size 32, symmetric) of the fp32 ONNX scorer.

Measured on onnx/validate.py (Spearman vs torch, 10 queries x 4 docs) - see README:
- Standard dynamic int8 (int8 activations per tensor) breaks this LLM completely (Spearman ~0), even when only
  the weight MatMuls are quantized and down_proj/o_proj are left in fp32: the activations after RMSNorm have
  outliers that a single per-tensor scale cannot represent.
- Weight-only block quantization keeps activations in float and quantizes weights in blocks of 32:
  8-bit gives Spearman 0.995 (used here), 4-bit 0.92-0.93 (below the 0.98 bar).
On an AVX2 CPU without VNNI these kernels are slower than fp32; on newer CPUs they can be faster - benchmark.
"""
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

M = Path(__file__).parent / "models"
src, out = M / "model_fp32.onnx", M / "model_int8.onnx"

model = onnx.load(str(src))
# accuracy_level=4: int8 compute inside each block; "/MatMul" is the tiny final yes-minus-no projection
q = MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True, accuracy_level=4, nodes_to_exclude=["/MatMul"])
q.process()
q.model.save_model_to_file(str(out), use_external_data_format=True)
print("quantized", out)
