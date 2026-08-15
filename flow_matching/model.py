"""小型 UNet：现代风格（时间调制 GroupNorm + SiLU），与 Wan/DiT 同款设计理念。

结构：32→16→8→4 四级，每级 num_res_blocks 个 ResBlock；
- 8×8 与 4×4 分辨率加 SelfAttention（全局依赖）
- 时间嵌入：正弦 → MLP → scale/shift（pre-modulation）
- 上采样路径与 down 严格镜像，skip 逐级拼接
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """时间 t -> 正弦嵌入（与 DDPM/DiT 一致）"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)      # [B, half]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.net[0].in_features))


class ResBlock(nn.Module):
    """GroupNorm + SiLU + Conv + 时间调制（scale/shift）。

    时间调制在 norm 之后、激活之前（pre-modulation，Wan/DiT 的 adaLN 思路）。
    """

    def __init__(self, cin: int, cout: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, cout)
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.modulation = nn.Linear(t_dim, cin * 2)          # scale, shift（作用在 norm1 输出上）
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(t_emb).chunk(2, dim=-1)  # [B, cin]
        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.drop(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)           # [B, HW, C]
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        h = F.scaled_dot_product_attention(q, k, v)           # 自动选 flash/math
        h = self.proj(h).transpose(1, 2).reshape(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    """一级 down：num_res_blocks 个 ResBlock（可选加 attn）+ 下采样"""

    def __init__(self, cin, cout, t_dim, num_blocks, use_attn, dropout):
        super().__init__()
        blocks = []
        cur = cin
        for i in range(num_blocks):
            blocks.append(ResBlock(cur, cout, t_dim, dropout))
            if use_attn:
                blocks.append(SelfAttention(cout))
            cur = cout
        self.blocks = nn.ModuleList(blocks)
        self.down = nn.Conv2d(cout, cout, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        skips = []
        for b in self.blocks:
            x = b(x, t_emb) if isinstance(b, ResBlock) else b(x)
            if isinstance(b, ResBlock):
                skips.append(x)
        return self.down(x), skips


class UpBlock(nn.Module):
    """一级 up：上采样 + num_blocks 个 ResBlock（每个都 concat 一个 skip）

    cin_first = below_ch + ch（第一个 block 吃上采样输入 + 本层第一个 skip）
    之后每个 block 输入 = 上一 block 输出(ch) + 下一个 skip(ch)
    """

    def __init__(self, below_ch, ch, t_dim, num_blocks, use_attn, dropout):
        super().__init__()
        blocks = []
        cur = below_ch + ch
        for _ in range(num_blocks):
            blocks.append(ResBlock(cur, ch, t_dim, dropout))
            if use_attn:
                blocks.append(SelfAttention(ch))
            cur = ch + ch
        self.blocks = nn.ModuleList(blocks)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t_emb, skips):
        x = self.up(x)
        for b in self.blocks:
            if isinstance(b, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = b(x, t_emb)
            else:
                x = b(x)
        return x


class UNet(nn.Module):
    """mini UNet：time-conditional，CIFAR-10 (32×32×3)。"""

    def __init__(self, cfg):
        super().__init__()
        t_dim = cfg.base_channels * 4
        self.t_mlp = TimestepMLP(cfg.base_channels, t_dim)

        chs = [cfg.base_channels * m for m in cfg.channel_mult]
        res_hw = [32 // (2 ** i) for i in range(len(chs))]
        self.conv_in = nn.Conv2d(cfg.in_channels, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        for level, ch in enumerate(chs):
            cin = chs[level - 1] if level > 0 else chs[0]
            self.downs.append(DownBlock(
                cin, ch, t_dim, cfg.num_res_blocks,
                use_attn=res_hw[level] in cfg.attn_resolutions,
                dropout=cfg.dropout))

        self.mid = nn.ModuleList([
            ResBlock(chs[-1], chs[-1], t_dim, cfg.dropout),
            SelfAttention(chs[-1]),
            ResBlock(chs[-1], chs[-1], t_dim, cfg.dropout),
        ])

        self.ups = nn.ModuleList()
        rev = list(reversed(chs))
        for level, ch in enumerate(rev):
            below_ch = rev[level - 1] if level > 0 else chs[-1]
            self.ups.append(UpBlock(
                below_ch, ch, t_dim, cfg.num_res_blocks,
                use_attn=res_hw[len(chs) - 1 - level] in cfg.attn_resolutions,
                dropout=cfg.dropout))

        self.out = nn.Sequential(
            nn.GroupNorm(32, chs[0]), nn.SiLU(),
            nn.Conv2d(chs[0], cfg.in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(t)
        h = self.conv_in(x)
        skips_total = []
        for d in self.downs:
            h, skips = d(h, t_emb)
            skips_total.extend(skips)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for u in self.ups:
            h = u(h, t_emb, skips_total)
        return self.out(h)
