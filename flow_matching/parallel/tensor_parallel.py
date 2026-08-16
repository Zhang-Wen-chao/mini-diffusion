"""张量并行层（移植自 mini-megatron + DiT 适配）

ColumnParallelLinear：按输出维切（QKV / FFN 前向投影）
RowParallelLinear：按输入维切 + 输出 all-reduce（attention 输出 / FFN 输出）

注意：DiT 的 adaLN 调制层（scale/shift）是**跨 rank 复制**的，
因为调制作用于每个 rank 本地已切分的 hidden 维度。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .all_reduce import all_reduce
from .process_groups import get_model_parallel


class ColumnParallelLinear(nn.Module):
    """按输出维切分（每 rank 存 out/tp 行）"""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """按输入维切分（每 rank 存 in/tp 列），输出 all-reduce 合并"""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        mpu = get_model_parallel()
        # bias 必须在 all-reduce 之后加：否则 bias 会被重复 tp 次
        output = F.linear(x, self.weight, None)
        output = all_reduce(output, mpu["tp_group"] if mpu else None)
        if self.bias is not None:
            output = output + self.bias
        return output
