



# ...existing code...
import onnx
import numpy as np
from onnx import numpy_helper
from typing import Tuple

from aicmder import model_info

model_path = "/home/faith/Yolov11-ONNX-Object-Detection/models/yolov11n.onnx"
model_path = "/home/faith/Yolo-FastestV2/yolo-fastestv2.onnx"
# model_path = "/home/faith/ONNX-YOLOv10-Object-Detection/models/yolov10n.onnx"
# model_path = "/home/faith/yolov5/yolov5n6-6.2-640.onnx"
# model_path = "/home/faith/yolov5/yolov5n6-6.2-320.onnx"
# model_path = "/home/faith/yolov5/yolov5n6-6.2-192.onnx"
# model_path = '/home/faith/yolov5/exp4/weights/best.onnx'
model_path = '/home/faith/yolov5/exp4/weights/full_best.onnx'
model_path = '/home/faith/yolov5/exp4/weights/transpose_best.onnx'
model_path = '/home/faith/yolov5/yolov8n.onnx'
model_path = '/home/faith/FastestDet/example/onnx-runtime/FastestDet.onnx'
def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception:
        return model

def shape_map_from_model(model: onnx.ModelProto):
    sm = {}
    graph = model.graph
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            shape_proto = vi.type.tensor_type.shape
            dims = []
            for d in shape_proto.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    dims.append(None)
            sm[vi.name] = dims
    # initializers also provide shapes
    for init in graph.initializer:
        sm[init.name] = list(init.dims)
    return sm

def get_initializer_dict(model: onnx.ModelProto):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

def compute_flops(model_path: str, input_shape=(1,3,320,320)) -> Tuple[int,int]:
    model = onnx.load(model_path)
    model = infer_shapes(model)
    smap = shape_map_from_model(model)
    inits = get_initializer_dict(model)

    total_macs = 0          # 用于 Conv/MatMul/Gemm 等按 MACs 计数的操作
    extra_flops = 0         # 用于按 element-count 计数的逐元素/激活等操作（直接为 FLOPs）
    per_node = []

    # 每个常见算子的每元素 FLOPs 估计（可根据需要调整）
    per_elem_cost = {
        "Add": 1,
        "Sub": 1,
        "Mul": 1,
        "Div": 4,               # 近似
        "Relu": 1,
        "LeakyRelu": 1,
        "Sigmoid": 4,           # 近似 (exp + div...)
        "Tanh": 4,
        "BatchNormalization": 2, # scale + shift (inference)
        "Sqrt": 4,
        "Exp": 4,
        "Log": 4,
        "Softmax": 6,           # 近似：exp + sum + div
        "GlobalAveragePool": 1, # average = adds / n -> count as 1 per output approx
        "AveragePool": 1,
        "MaxPool": 0,           # 比较, 可视为 0 FLOPs 或 1
        "Resize": 1,            # 近似插值成本
        "Concat": 0,
        "Transpose": 0,
        "Reshape": 0,
        "SiLU": 5, # ✅ 新增：Sigmoid(≈4) + Mul(≈1) 
        "Swish": 5 # ✅ 有些模型用 Swish 表示 SiLU
    }

    for node in model.graph.node:
        op = node.op_type
        macs = 0

        inputs = list(node.input) # ✅ 转换成普通 list 
        outputs = list(node.output)

        if op == "Conv":
            if len(node.input) < 2:
                continue
            wname = node.input[1]
            if wname not in inits:
                continue
            w = inits[wname]  # (out_c, in_c/group, kh, kw)
            out_name = node.output[0] if node.output else None
            out_shape = smap.get(out_name)
            if out_shape is None or len(out_shape) < 4:
                continue
            N, Cout, Hout, Wout = out_shape[:4]
            _, Cin_per_group, kh, kw = w.shape
            group = 1
            for a in node.attribute:
                if a.name == "group":
                    group = a.i
            macs = (N if N else 1) * Cout * Hout * Wout * (Cin_per_group * kh * kw)
            # bias add: one add per output element -> count later as extra_flops if desired
            # 但这里保持 macs 为乘加计数，后面 total_flops = total_macs*2 + extra_flops

        elif op in ("MatMul",):
            a_name, b_name = (inputs + [None, None])[:2]
            ash = smap.get(a_name)
            bsh = smap.get(b_name)
            if not ash or not bsh:
                continue
            M = ash[-2]; K = ash[-1]; N = bsh[-1]
            batch = int(np.prod([d for d in ash[:-2] if d])) if len(ash) > 2 else 1
            if None in (M, K, N):
                continue
            macs = batch * M * N * K

        elif op in ("Gemm",):
            a_name, b_name = (node.input + [None, None])[:2]
            ash = smap.get(a_name)
            bsh = smap.get(b_name)
            if not ash or not bsh:
                continue
            M = ash[-2]; K = ash[-1]; N = bsh[-1]
            batch = int(np.prod([d for d in ash[:-2] if d])) if len(ash) > 2 else 1
            if None in (M, K, N):
                continue
            macs = batch * M * N * K

        elif op in ("ConvTranspose",):
            if len(node.input) < 2:
                continue
            wname = node.input[1]
            if wname not in inits:
                continue
            w = inits[wname]
            out_name = node.output[0] if node.output else None
            out_shape = smap.get(out_name)
            if out_shape is None or len(out_shape) < 4:
                continue
            N, Cout, Hout, Wout = out_shape[:4]
            _, Cin_per_group, kh, kw = w.shape
            macs = (N if N else 1) * Cout * Hout * Wout * (Cin_per_group * kh * kw)

        else:
            # 对于其它逐元素或激活类操作，按输出元素数量乘以每元素成本估算 FLOPs
            cost = per_elem_cost.get(op, None)
            if cost is not None:
                out_name = node.output[0] if node.output else None
                out_shape = smap.get(out_name)
                if out_shape:
                    # 计算输出元素数量（忽略 None 维）
                    elems = 1
                    unknown = False
                    for d in out_shape:
                        if d is None:
                            unknown = True
                            break
                        elems *= int(d)
                    if not unknown:
                        extra_flops += elems * cost
                        per_node.append((node.name or out_name, op, elems * cost))
                continue
            # 未列出的 op 将被跳过或可在此处扩展

        if macs:
            total_macs += macs
            per_node.append((node.name or node.output[0], op, macs))

    # MACs -> FLOPs (乘加计为 2 FLOPs/MAC)
    total_flops = total_macs * 2 + extra_flops
    return total_macs, total_flops, per_node

if __name__ == "__main__":
    total_macs, total_flops, per_node = compute_flops(model_path, input_shape=(1,3,320,320))
    print(f"Total MACs: {total_macs:,} ({total_macs/1e9:.3f} GMAC)")
    print(f"Estimated Total FLOPs: {total_flops:,} ({total_flops/1e9:.3f} GFLOPS)")
    # optional: print top nodes by MACs
    per_node_sorted = sorted(per_node, key=lambda x: x[2], reverse=True)[:10]
    for name, op, macs in per_node_sorted:
        print(f"{op} {name}: {macs:,}")


# model_path = "/home/faith/yolov5/yolov5n6-6.2-320.onnx" 
# model = onnx.load(model_path)

# # 统计 FLOPs 和参数量
# profile = onnx_tool.model_profile(model_path)
# print(profile)

