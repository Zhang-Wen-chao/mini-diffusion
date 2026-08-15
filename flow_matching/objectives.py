"""训练目标：DDPM（epsilon 预测）与 Flow Matching（velocity 预测）。

两者共用同一个模型（UNet），只差「加噪路径」和「预测对象」——
这正是 02/03 笔记的核心对比。
"""
from __future__ import annotations

import torch


class DDPMObjective:
    """前向: x_t = √ᾱ_t·x + √(1-ᾱ_t)·ε；目标: 预测 ε（Ho et al. 2020 简化损失）"""

    def __init__(self, cfg):
        if cfg.ddpm.schedule == "linear":
            betas = torch.linspace(cfg.ddpm.beta_start, cfg.ddpm.beta_end,
                                   cfg.ddpm.num_timesteps)
        else:  # cosine (Nichol & Dhariwal 2021)
            t = torch.arange(cfg.ddpm.num_timesteps + 1) / cfg.ddpm.num_timesteps
            alpha_bar = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
        self.num_timesteps = cfg.ddpm.num_timesteps
        self.alphas = 1 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        """x_t = √ᾱ_t·x + √(1-ᾱ_t)·ε"""
        a = self.alpha_bar.to(x.device)[t][:, None, None, None]
        eps = torch.randn_like(x)
        return a.sqrt() * x + (1 - a).sqrt() * eps, eps

    def target(self, x_t, t, eps):
        return eps

    def loss(self, model, x, t):
        x_t, eps = self.add_noise(x, t)
        eps_pred = model(x_t, t)
        return torch.nn.functional.mse_loss(eps_pred, eps)

    def predict_x0(self, x_t, t, eps_pred):
        a = self.alpha_bar.to(x_t.device)[t][:, None, None, None]
        return (x_t - (1 - a).sqrt() * eps_pred) / a.sqrt()

    def timesteps(self, batch_size, device, rng=None):
        gen = torch.Generator(device=device) if rng is None else None
        if gen is not None:
            gen.manual_seed(0)
        return torch.randint(0, self.num_timesteps, (batch_size,),
                             device=device, generator=gen)


class FlowObjective:
    """前向: x_t = (1-t)·z + t·x (z=噪声, x=数据)；目标: 预测速度 v = x - z。

    这就是 SD3 / Wan2.x / Flux 的直线插值路径 + velocity 预测（02 笔记）。
    t 采样策略：
      uniform     均匀（FM 原始）
      midpoint    t=0.5 恒采样（工程 trick，均匀且高信息量）
      logit_normal SD3 用，偏向中间信噪比
    """

    def __init__(self, cfg):
        self.t_sampler = cfg.flow_t_sampler

    def sample_t(self, batch_size, device):
        if self.t_sampler == "midpoint":
            return torch.full((batch_size,), 0.5, device=device)
        if self.t_sampler == "logit_normal":
            from torch.distributions import Normal
            return torch.sigmoid(Normal(0.0, 1.0).sample((batch_size,)).to(device))
        return torch.rand(batch_size, device=device)

    def add_noise(self, x, t, z=None):
        """直线插值：x_t = (1-t)·z + t·x"""
        t = t[:, None, None, None]
        if z is None:
            z = torch.randn_like(x)
        return (1 - t) * z + t * x, z

    def velocity(self, x, z):
        """v = d x_t/dt = x - z（直线路径的速度与 t 无关）"""
        return x - z

    def loss(self, model, x, t):
        x_t, z = self.add_noise(x, t)
        v = self.velocity(x, z)
        v_pred = model(x_t, t)
        return torch.nn.functional.mse_loss(v_pred, v)

    def timesteps(self, batch_size, device, rng=None):
        return self.sample_t(batch_size, device)


def build_objective(cfg):
    return DDPMObjective(cfg) if cfg.objective == "ddpm" else FlowObjective(cfg)
