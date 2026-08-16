"""DiT：Diffusion Transformer（Peebles & Xie 2023，TP 友好版）

与 UNet 的区别（架构演进，04 笔记）：
- patchify：图像切 patch → token 序列（标准 transformer 输入）
- adaLN-zero：时间条件通过 scale/shift 调制每个 block 的 norm，
  初始化为零（残差分支从恒等开始，训练稳定）
- 无卷积下采样/跳跃连接：纯 transformer block

TP 切分（与 GPT 同构）：
- QKV 投影：ColumnParallelLinear（按 head 维切，每个 rank 处理部分 head）
- Attention 输出：RowParallelLinear（合并 + all-reduce）
- FFN 上投影：Column；FFN 下投影：Row
- adaLN 调制层、patchify、pos_embed：跨 rank 复制（无通信）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import sinusoidal_embedding
from .parallel.process_groups import get_model_parallel
from .parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class TimestepEmbedder(nn.Module):
    """t -> 正弦嵌入 -> MLP（输出调制向量）"""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.frequency_embedding_size))


class Attention(nn.Module):
    """自注意力（单卡版用普通 Linear；to_tp() 统一替换为 TP 版）

    TP 下每个 rank 处理 heads/tp 个 head：Column 切分按连续维，
    q/k/v 各取 D/tp 连续段 → reshape 后正好是前 heads/tp 个 head。
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)                                   # [B, N, 3D] (或 3D/tp)
        qkv = qkv.reshape(B, N, 3, -1, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # [B, heads, N, hd]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class DiTBlock(nn.Module):
    """adalN-zero + attention + FFN"""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        # adaLN-zero：6 路调制，初始化为零（残差从恒等开始）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """adalN-zero + Linear -> 像素空间"""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(DiTBlock.modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """mini DiT：patchify + N×DiTBlock + unpatchify（CIFAR 32×32）

    配置对齐 DiT-S：hidden 384, depth 6, heads 6, patch 4 → ~33M
    TP 版：构造后调用 to_tp()（attention/FFN/Final 的 Linear 换 Parallel 版）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.image_size = cfg.image_size
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.n_patches = n_patches

        self.x_embedder = nn.Conv2d(
            cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size,
            stride=cfg.patch_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_size, cfg.num_heads, mlp_ratio=cfg.mlp_ratio)
            for _ in range(cfg.depth)])
        self.final = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        h = self.x_embedder(x).flatten(2).transpose(1, 2)   # [B, N, D]
        h = h + self.pos_embed
        c = self.t_embedder(t)                              # [B, D]
        for block in self.blocks:
            h = block(h, c)
        h = self.final(h, c)                                # [B, N, p²C]
        # unpatchify: [B, H/p, W/p, p, p, C] -> [B, C, H, W]
        h = h.reshape(B, self.image_size // p, self.image_size // p, p, p,
                      self.in_channels)
        h = h.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.in_channels, self.image_size, self.image_size)
        return h

    def to_tp(self):
        """把 attention/FFN/Final 的 Linear 替换为 TP 版（就地，返回 self）

        前置：所有 rank 已持有完整权重（train_dist 里先 broadcast）。
        拷贝时按 tp_rank 切分：Column 按输出维取连续行，Row 按输入维取连续列。
        """
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        if tp_size <= 1:
            return self
        tp_rank = mpu["tp_rank"]

        def _to_col(lin):
            out_per = lin.out_features // tp_size
            new = ColumnParallelLinear(lin.in_features, lin.out_features,
                                       bias=lin.bias is not None)
            sl = slice(tp_rank * out_per, (tp_rank + 1) * out_per)
            new.weight.data.copy_(lin.weight.data[sl])
            if lin.bias is not None:
                new.bias.data.copy_(lin.bias.data[sl])
            return new

        def _to_row(lin):
            in_per = lin.in_features // tp_size
            new = RowParallelLinear(lin.in_features, lin.out_features,
                                    bias=lin.bias is not None)
            sl = slice(tp_rank * in_per, (tp_rank + 1) * in_per)
            new.weight.data.copy_(lin.weight.data[:, sl])
            if lin.bias is not None:
                new.bias.data.copy_(lin.bias.data)
            return new

        for block in self.blocks:
            block.attn.qkv = _to_col(block.attn.qkv)
            block.attn.proj = _to_row(block.attn.proj)
            block.mlp[0] = _to_col(block.mlp[0])
            block.mlp[2] = _to_row(block.mlp[2])
        # FinalLayer 不参与 TP：其输入是最后一个 block 的完整 hidden（Row 已合并）
        # 新构造的 Parallel 层权重在 CPU，回迁到模型所在 device
        self = self.to(next(self.parameters()).device)
        # 记录切分层参数：未切分层（adaLN/norm/patchify/pos_embed/t_embedder/final）
        # 的梯度在每 rank 都完整计算，需要除以 tp_size（Megatron 规则）
        sharded = set()
        for name, _ in self.named_parameters():
            if any(mark in name for mark in
                   (".attn.qkv.", ".attn.proj.", ".mlp.0.", ".mlp.2.")):
                sharded.add(name)
        self._tp_sharded = sharded
        return self
