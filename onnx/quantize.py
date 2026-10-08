"""Dynamic int8 quantization of the fp32 ONNX reranker."""
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

M = Path(__file__).parent / "models"
src = M / "model_fp32.onnx"
pre = M / "pre" / "model.onnx"
pre.parent.mkdir(exist_ok=True)
quant_pre_process(str(src), str(pre), skip_symbolic_shape=True, save_as_external_data=True,
                  all_tensors_to_one_file=True, external_data_location="model.onnx.data")
print("preprocessed")
out = M / "model_int8.onnx"
quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8, per_channel=True,
                 use_external_data_format=True)
print("quantized", out)
