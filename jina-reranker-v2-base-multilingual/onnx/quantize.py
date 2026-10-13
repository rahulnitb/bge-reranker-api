"""Dynamic int8 quantization (per-channel weights) of the fused ONNX graph - feed-forward MatMuls only.

Measured on onnx/validate.py: quantizing everything (incl. the fused Attention op's QKV projection) drops Spearman
vs torch to 0.83 for this model; leaving Attention and the embedding Gather in fp32 keeps 0.987 at the same speed
(the int8 speedup comes from the MatMuls). The cost is file size: ~920 MB instead of ~280 MB.
"""
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

M = Path(__file__).parent / "models"
src, out = M / "model_opt.onnx", M / "model_int8.onnx"
quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8, per_channel=True,
                 op_types_to_quantize=["MatMul"],
                 # fused contrib ops (SkipLayerNorm, BiasGelu, ...) defeat shape inference; their outputs are float
                 extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
print("quantized", out)
