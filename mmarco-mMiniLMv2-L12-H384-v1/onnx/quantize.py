"""Dynamic int8 quantization (per-channel weights) of the fused ONNX graph."""
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

M = Path(__file__).parent / "models"
src, out = M / "model_opt.onnx", M / "model_int8.onnx"
quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8, per_channel=True,
                 # fused contrib ops (SkipLayerNorm, BiasGelu, ...) defeat shape inference; their outputs are float
                 extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
print("quantized", out)
