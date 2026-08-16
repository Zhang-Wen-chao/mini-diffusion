"""MoE DiT：双 FFN 专家 + timestep 路由（Wan2.2 范式迷你版）

对齐 Wan2.2 的核心思想：**按噪声水平分工**——
- t >= boundary（高噪声）：高噪声专家（学整体布局）
- t <  boundary（低噪声）：低噪声专家（学细节）
attention / adaLN / patchify 共享，只有 FFN 专家化。

与 Wan2.2 的差异（教学简化）：
- Wan2.2 是两个完整模型分开训练；我们是单模型双专家联合训练
  （batch 内 t 混合，用 mask 分流）

参数量：dense 的 (1 + 1/2·mlp 占比) ≈ 1.4x；**激活参数量与 dense 相同**。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .dit import Attention, FinalLayer, TimestepEmbedder


class MoEFeedForward(nn.Module):
    """双专家 FFN：expert_hi（高噪声）+ expert_lo（低噪声）"""

    def __init__(self, hidden_size: int, mlp_hidden: int, bias: bool = True):
        super().__init__()
        self.expert_hi = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=bias),
        )
        self.expert_lo = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=bias),
        )

    def forward(self, x: torch.Tensor, is_high: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D], is_high: [B] bool"""
        hi = self.expert_hi(x)
        lo = self.expert_lo(x)
        mask = is_high[:, None, None].float()
        return hi * mask + lo * (1 - mask)


class MoEDiTBlock(nn.Module):
    """adaLN-zero + attention + 双专家 FFN"""

    def __init__(self, hidden_size: int, num_heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MoEFeedForward(hidden_size, mlp_hidden)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, is_high):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1), is_high)
        return x


class MoEDiT(nn.Module):
    """MoE 版 DiT（与 DiT 同构，FFN 双专家化）"""

    def __init__(self, cfg: ModelConfig, boundary: float = 0.5):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.image_size = cfg.image_size
        self.boundary = boundary
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.n_patches = n_patches

        self.x_embedder = nn.Conv2d(
            cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size,
            stride=cfg.patch_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            MoEDiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio)
            for _ in range(cfg.depth)])
        self.final = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        h = self.x_embedder(x).flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        c = self.t_embedder(t)
        is_high = t >= self.boundary                    # [B] 高噪声路由
        for block in self.blocks:
            h = block(h, c, is_high)
        h = self.final(h, c)
        h = h.reshape(B, self.image_size // p, self.image_size // p, p, p,
                      self.in_channels)
        h = h.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.in_channels, self.image_size, self.image_size)
        return h
