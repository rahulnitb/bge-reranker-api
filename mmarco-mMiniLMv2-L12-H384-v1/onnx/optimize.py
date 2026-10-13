"""Fuse transformer subgraphs (Attention, SkipLayerNorm, BiasGelu, ...) with the ONNX Runtime transformer optimizer."""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.transformers import optimizer

M = Path(__file__).parent / "models"
src, out = M / "model_fp32.onnx", M / "model_opt.onnx"

# Eager attention computes Softmax(Mul(QK^T, 1/sqrt(d)) + Mul(Sub(1, mask), -10000)). The optimizer finds both
# the score scaling and the mask as "the first Mul input of that Add", so the two collide and attention never
# fuses. Rewrite the scaling as the equivalent Div(QK^T, sqrt(d)) - the classic BERT pattern it expects.
m = onnx.load(str(src))
producer = {o: n for n in m.graph.node for o in n.output}
consts = {n.output[0]: n for n in m.graph.node if n.op_type == "Constant"}
softmax_inputs = {n.input[0] for n in m.graph.node if n.op_type == "Softmax"}
rewritten = 0
for add in (n for n in m.graph.node if n.op_type == "Add" and n.output[0] in softmax_inputs):
    for name in add.input:
        mul = producer.get(name)
        if mul is None or mul.op_type != "Mul" or producer.get(mul.input[0], mul).op_type != "MatMul":
            continue
        c = consts.get(mul.input[1])
        if c is None:
            continue
        scale = numpy_helper.to_array(c.attribute[0].t)
        c.attribute[0].t.CopyFrom(numpy_helper.from_array((1.0 / scale).astype(np.float32), c.attribute[0].t.name))
        mul.op_type = "Div"
        rewritten += 1
print("attention scale Mul -> Div:", rewritten)

# Q/K/V are reshaped with view(batch, seq, -1, head_dim); the optimizer reads num_heads from that "-1" and
# aborts. Pin it to the real value (12 heads x 32 = 384, so the reshape result is identical).
m.graph.initializer.append(numpy_helper.from_array(np.array([12], dtype=np.int64), "num_heads_12"))
pinned = 0
for n in m.graph.node:
    if n.op_type == "Concat" and len(n.input) == 4 and n.input[2] in consts and n.input[3] in consts:
        if (numpy_helper.to_array(consts[n.input[2]].attribute[0].t).tolist() == [-1]
                and numpy_helper.to_array(consts[n.input[3]].attribute[0].t).tolist() == [32]):
            n.input[2] = "num_heads_12"
            pinned += 1
print("pinned num_heads in reshapes:", pinned)

# XLM-RoBERTa is optimized with the "bert" graph patterns; 12 heads x 32 dims = hidden 384
opt = optimizer.optimize_model(m, model_type="bert", num_heads=12, hidden_size=384, opt_level=1)
print("fused ops:", {k: v for k, v in opt.get_fused_operator_statistics().items() if v})
opt.save_model_to_file(str(out))
print("saved", out)
