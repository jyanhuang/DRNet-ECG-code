# models/sparse_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv1d(nn.Module):
    def __init__(self, out_ch, in_ch, ks, weight1d, bias,
                 idx0, idx1, idx2,
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.out_channels = out_ch
        self.in_channels  = in_ch
        self.kernel_size  = ks
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups   = groups

        self.register_parameter('weight_pruned', nn.Parameter(weight1d))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.register_buffer('idx0', idx0)
        self.register_buffer('idx1', idx1)
        self.register_buffer('idx2', idx2)

    def forward(self, x):
        W = torch.zeros(self.out_channels, self.in_channels, self.kernel_size,
                        dtype=self.weight_pruned.dtype,
                        device=self.weight_pruned.device)
        W[self.idx0, self.idx1, self.idx2] = self.weight_pruned
        return F.conv1d(x, W, self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, weight1d, bias, row_idx, col_idx):
        super().__init__()
        self.in_features  = in_f
        self.out_features = out_f
        self.register_parameter('weight_pruned', nn.Parameter(weight1d))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.register_buffer('row_idx', row_idx)
        self.register_buffer('col_idx', col_idx)

    def forward(self, x):
        W = torch.zeros(self.out_features, self.in_features,
                        dtype=self.weight_pruned.dtype,
                        device=self.weight_pruned.device)
        W[self.row_idx, self.col_idx] = self.weight_pruned
        return F.linear(x, W, self.bias)