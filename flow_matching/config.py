"""配置：dataclass + CLI 解析，单一入口"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    in_channels: int = 3          # CIFAR-10 RGB
    base_channels: int = 128      # 第一层通道数
    channel_mult: tuple = (1, 2, 2, 2)   # 32→16→8→4
    num_res_blocks: int = 2
    attn_resolutions: tuple = (8, 4)     # 低分辨率加 attention
    dropout: float = 0.0


@dataclass
class DDPMConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"      # linear | cosine


@dataclass
class TrainConfig:
    objective: str = "flow"       # ddpm | flow
    data_dir: str = "/path/to/mini-diffusion-test/data"
    out_dir: str = "/path/to/mini-diffusion-test/runs/phase2"
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 50000
    log_every: int = 100
    ckpt_every: int = 5000
    sample_every: int = 5000
    mixed_precision: str = "bf16"     # none | fp16 | bf16
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    seed: int = 0
    flow_t_sampler: str = "uniform"   # uniform | logit_normal | midpoint
    ddim_steps: int = 50              # ddpm 目标的采样步数
    euler_steps: int = 50             # flow 目标的采样步数
    resume: str | None = None         # checkpoint 路径
    model: ModelConfig = field(default_factory=ModelConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["ddpm", "flow"], default="flow")
    ap.add_argument("--data-dir", default=TrainConfig.data_dir)
    ap.add_argument("--out-dir", default=TrainConfig.out_dir)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")
    ap.add_argument("--flow-t-sampler", choices=["uniform", "logit_normal", "midpoint"], default="uniform")
    ap.add_argument("--euler-steps", type=int, default=50)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint 路径")
    args = ap.parse_args()

    cfg = TrainConfig(
        objective=args.objective, data_dir=args.data_dir, out_dir=args.out_dir,
        batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps,
        mixed_precision=args.mixed_precision, flow_t_sampler=args.flow_t_sampler,
        euler_steps=args.euler_steps, ddim_steps=args.ddim_steps, seed=args.seed,
        resume=args.resume,
    )
    return cfg
