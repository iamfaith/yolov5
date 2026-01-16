import torch

class GridTorch:
    def __init__(self, anchors, stride):
        # anchors: list of tensors, shape (na, 2) per layer
        # stride: list of ints
        self.anchors = anchors
        self.stride = stride
        self.nl = len(stride)   # number of layers
        self.na = anchors[0].shape[0]

        # 归一化 anchors
        self.anchors = [a / s for a, s in zip(self.anchors, self.stride)]

    def make_grid(self, nx=20, ny=20, i=0):
        shape = (1, self.na, ny, nx, 2)
        y = torch.arange(ny, dtype=torch.float32)
        x = torch.arange(nx, dtype=torch.float32)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

import numpy as np

class GridNumpy:
    def __init__(self, anchors, stride):
        # anchors: list of ndarrays, shape (na, 2) per layer
        # stride: list of ints
        self.anchors = anchors
        self.stride = stride
        self.nl = len(stride)
        self.na = anchors[0].shape[0]

        # 归一化 anchors
        self.anchors = [a / s for a, s in zip(self.anchors, self.stride)]

    def make_grid(self, nx=20, ny=20, i=0):
        shape = (1, self.na, ny, nx, 2)
        y = np.arange(ny, dtype=np.float32)
        x = np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        grid = np.stack((xv, yv), axis=2)
        grid = np.broadcast_to(grid, shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.broadcast_to(anchor_grid, shape)
        return grid, anchor_grid



# 假设 anchors 和 stride
anchors_torch = [
    torch.tensor([[10., 13.], [16., 30.], [33., 23.]], dtype=torch.float32),
    torch.tensor([[30., 61.], [62., 45.], [59., 119.]], dtype=torch.float32),
    torch.tensor([[116., 90.], [156., 198.], [373., 326.]], dtype=torch.float32)
]
stride_torch = [8, 16, 32]

anchors_numpy = [a.numpy() for a in anchors_torch]
stride_numpy = stride_torch
print(anchors_numpy)
gt = GridTorch(anchors_torch, stride_torch)
gn = GridNumpy(anchors_numpy, stride_numpy)

grid_t, anchor_t = gt.make_grid(nx=5, ny=5, i=0)
grid_n, anchor_n = gn.make_grid(nx=5, ny=5, i=0)

print("Torch grid sample:\n", grid_t[0,0,:3,:3,:])
print("Numpy grid sample:\n", grid_n[0,0,:3,:3,:])

print("\nTorch anchor_grid sample:\n", anchor_t[0,:,0,0,:])
print("Numpy anchor_grid sample:\n", anchor_n[0,:,0,0,:])
