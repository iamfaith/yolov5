import onnx
import numpy as np

def compute_onnx_flops(model_path, input_shape=(1, 3, 640, 640)):
    model = onnx.load(model_path)
    graph = model.graph

    shape_map = {}
    for inp in graph.input:
        dims = [d.dim_value if (d.dim_value > 0) else None for d in inp.type.tensor_type.shape.dim]
        shape_map[inp.name] = dims

    total_macs = 0
    per_node = []

    for node in graph.node:
        op = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)

        # Conv
        if op == "Conv":
            x = shape_map.get(inputs[0])
            w = shape_map.get(inputs[1])
            if x is None or w is None:
                continue
            batch, Cin, H, W = x
            Cout, _, kH, kW = w
            out_h = H - kH + 1
            out_w = W - kW + 1
            macs = batch * Cout * out_h * out_w * Cin * kH * kW
            total_macs += macs
            per_node.append((node.name, macs))

        # MatMul
        elif op == "MatMul":
            a = shape_map.get(inputs[0])
            b = shape_map.get(inputs[1])
            if a is None or b is None:
                continue
            M, K = a[-2], a[-1]
            N = b[-1]
            batch = int(np.prod([d for d in a[:-2] if d])) if len(a) > 2 else 1
            if None in (M, K, N):
                continue
            macs = batch * M * N * K
            total_macs += macs
            per_node.append((node.name, macs))

        # Gemm
        elif op == "Gemm":
            a = shape_map.get(inputs[0])
            b = shape_map.get(inputs[1])
            if a is None or b is None:
                continue
            M, K = a[-2], a[-1]
            N = b[-1]
            if None in (M, K, N):
                continue
            macs = M * N * K
            total_macs += macs
            per_node.append((node.name, macs))

        # BatchNorm
        elif op == "BatchNormalization":
            x = shape_map.get(inputs[0])
            if x is None:
                continue
            macs = np.prod([d for d in x if d])
            total_macs += macs
            per_node.append((node.name, macs))

        # Add / Mul / Sub / Div
        elif op in ("Add", "Mul", "Sub", "Div"):
            a = shape_map.get(inputs[0])
            if a is None:
                continue
            macs = np.prod([d for d in a if d])
            total_macs += macs
            per_node.append((node.name, macs))

        # 激活函数
        elif op in ("Relu", "Sigmoid", "Tanh", "LeakyRelu"):
            x = shape_map.get(inputs[0])
            if x is None:
                continue
            macs = np.prod([d for d in x if d])
            total_macs += macs
            per_node.append((node.name, macs))

        # SiLU (Swish: x * sigmoid(x))
        elif op in ("SiLU", "Swish"):
            x = shape_map.get(inputs[0])
            if x is None:
                continue
            macs = 2 * np.prod([d for d in x if d])  # sigmoid + 乘法
            total_macs += macs
            per_node.append((node.name, macs))

        # Concat / Upsample / Resize 忽略
        elif op in ("Concat", "Upsample", "Resize"):
            continue

        else:
            continue

    total_flops = 2 * total_macs
    gflops = total_flops / 1e9
    return gflops, per_node


if __name__ == "__main__":
    model_path = "/home/faith/ONNX-YOLOv10-Object-Detection/models/yolov10n.onnx"
    gflops, per_node = compute_onnx_flops(model_path, input_shape=(1, 3, 640, 640))
    print(f"Total GFLOPs: {gflops:.2f}")
    for name, macs in per_node[:10]:
        print(f"{name}: {macs/1e6:.2f} M MACs")
