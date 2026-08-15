"""采样器：DDPM ancestral / DDIM 确定性 / flow 欧拉。

关键理解（03 笔记）：所有采样都是解 ODE——
- DDPM 目标: 去噪 + 加回小噪声（ancestral）
- DDIM: 确定性 ODE 离散化，可跳步
- flow: 欧拉法解 dx/dt = v_θ，步数越少越依赖路径"直"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_ddpm_ancestral(model, objective, shape, steps=1000, device="cuda"):
    """标准 DDPM 采样（T 步逐步去噪）"""
    x = torch.randn(shape, device=device)
    for i in reversed(range(objective.num_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        a = objective.alphas.to(device)[i]
        ab = objective.alpha_bar.to(device)[i]
        ab_prev = objective.alpha_bar.to(device)[i - 1] if i > 0 else torch.ones((), device=device)
        x0_pred = (x - (1 - ab).sqrt() * eps_pred) / a.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        if i > 0:
            # 后验 q(x_{t-1}|x_t,x_0)
            beta = 1 - a
            mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0_pred + \
                   (a * (1 - ab_prev) / (1 - ab)).sqrt() * x
            x = mean + torch.randn_like(x) * (beta * (1 - ab_prev) / (1 - ab)).sqrt()
        else:
            x = x0_pred
    return x


@torch.no_grad()
def sample_ddim(model, objective, shape, steps=50, eta=0.0, device="cuda"):
    """DDIM 确定性采样（可跳步），eta=0 即 PF-ODE 离散化"""
    x = torch.randn(shape, device=device)
    ts = torch.linspace(objective.num_timesteps - 1, 0, steps, device=device).long()
    ts_prev = torch.cat([ts[1:], torch.zeros(1, dtype=torch.long, device=device)])
    for i, t in enumerate(ts):
        t_b = t.expand(shape[0])
        eps_pred = model(x, t_b)
        ab = objective.alpha_bar.to(device)[t]
        ab_prev = objective.alpha_bar.to(device)[ts_prev[i]]
        x0_pred = (x - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        dir_xt = (1 - ab_prev - eta ** 2 * (1 - ab)).sqrt() * eps_pred
        x = ab_prev.sqrt() * x0_pred + dir_xt
        if eta > 0:
            x = x + eta * torch.randn_like(x) * (1 - ab).sqrt()
    return x


@torch.no_grad()
def sample_euler(model, objective, shape, steps=50, device="cuda"):
    """flow 目标：欧拉法解 dx/dt = v_θ，从纯噪声(0)走到数据(1)"""
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), (i + 0.5) * dt, device=device)  # midpoint 规则
        v = model(x, t)
        x = x + dt * v
    return x


def sample(model, objective, cfg, n=64, device="cuda", steps=None):
    shape = (n, cfg.model.in_channels, 32, 32)
    if cfg.objective == "ddpm":
        return sample_ddim(model, objective, shape, steps=steps or cfg.ddim_steps, device=device)
    return sample_euler(model, objective, shape, steps=steps or cfg.euler_steps, device=device)
