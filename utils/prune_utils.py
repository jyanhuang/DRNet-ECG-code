import torch
import torch.nn as nn
# ------- 下面这一段就是缺失的 _get_parent_and_attr -------
def _get_parent_and_attr(root, name):
    names = name.split('.')
    parent = root
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]
# 稀疏层定义（直接拷贝自训练代码）
class MaskedConv1d(nn.Module):
    def __init__(self, out_ch, in_ch, ks, weight1d, bias,
                 idx0, idx1, idx2, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.out_channels, self.in_channels = out_ch, in_ch
        self.kernel_size = ks
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.register_parameter('weight_pruned', nn.Parameter(weight1d))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.register_buffer('idx0', idx0)
        self.register_buffer('idx1', idx1)
        self.register_buffer('idx2', idx2)



    def forward(self, x):
        W = torch.zeros(self.out_channels, self.in_channels, self.kernel_size,
                        dtype=self.weight_pruned.dtype, device=self.weight_pruned.device)
        W[self.idx0, self.idx1, self.idx2] = self.weight_pruned
        return nn.functional.conv1d(x, W, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)


class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, weight1d, bias, row_idx, col_idx):
        super().__init__()
        self.in_features, self.out_features = in_f, out_f
        self.register_parameter('weight_pruned', nn.Parameter(weight1d))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.register_buffer('row_idx', row_idx)
        self.register_buffer('col_idx', col_idx)

    def forward(self, x):
        W = torch.zeros(self.out_features, self.in_features,
                        dtype=self.weight_pruned.dtype, device=self.weight_pruned.device)
        W[self.row_idx, self.col_idx] = self.weight_pruned
        return nn.functional.linear(x, W, self.bias)


# 把普通模型“还原”成剪枝模型
def rebuild_pruned_model(model, state_dict):
    """
    model: 普通 HeartNet/ResNet1D ...（结构同训练前）
    state_dict: 剪枝后保存的 state_dict（含 weight_pruned / idx）
    return: 已经替换好稀疏层的模型（可立即 load_state_dict）
    """
    # 先加载，让模型创建好权重形状
    model.load_state_dict(state_dict, strict=False)

    # 逐层替换
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # 从 state_dict 拿稀疏参数
            w1d   = state_dict[f"{name}.weight_pruned"]
            bias  = state_dict.get(f"{name}.bias", None)
            rows  = state_dict[f"{name}.row_idx"]
            cols  = state_dict[f"{name}.col_idx"]
            new_layer = MaskedLinear(m.in_features, m.out_features, w1d, bias, rows, cols)
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_layer)

        elif isinstance(m, nn.Conv1d):
            w1d  = state_dict[f"{name}.weight_pruned"]
            bias = state_dict.get(f"{name}.bias", None)
            idx0 = state_dict[f"{name}.idx0"]
            idx1 = state_dict[f"{name}.idx1"]
            idx2 = state_dict[f"{name}.idx2"]
            new_layer = MaskedConv1d(
                m.out_channels, m.in_channels, m.kernel_size[0],
                w1d, bias, idx0, idx1, idx2,
                stride=m.stride[0], padding=m.padding[0],
                dilation=m.dilation[0], groups=m.groups
            )
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_layer)

    return model