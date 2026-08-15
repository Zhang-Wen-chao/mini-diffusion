"""训练入口：单卡，DDPM / Flow 双目标可选。

用法：
  python3 -m flow_matching.train --objective flow    --max-steps 30000
  python3 -m flow_matching.train --objective ddpm    --max-steps 30000
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from .config import parse_args
from .data import get_cifar10_loader
from .model import UNet
from .objectives import build_objective
from .sampler import sample


class EMA:
    """指数滑动平均：采样用 EMA 权重（标准做法，稳采样质量）"""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(cfg.model).to(device)
    objective = build_objective(cfg)
    print(f"[cfg] objective={cfg.objective} t_sampler={cfg.flow_t_sampler} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ema = EMA(model, cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 5000))          # warmup

    step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow = ckpt["ema"]
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]

    use_amp = cfg.mixed_precision != "none"
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision == "fp16")

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_dl = get_cifar10_loader(cfg.batch_size, cfg.data_dir)
    model.train()
    t0 = time.time()
    it = iter(train_dl)

    while step < cfg.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)

        x = batch.to(device)
        t = objective.timesteps(x.shape[0], device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = objective.loss(model, x, t)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        if step % cfg.log_every == 0:
            sps = (cfg.log_every * cfg.batch_size) / (time.time() - t0)
            t0 = time.time()
            print(f"step {step:6d} | loss {loss.item():.4f} | "
                  f"{sps:6.0f} img/s | lr {scheduler.get_last_lr()[0]:.2e}")

        if step % cfg.sample_every == 0:
            ema.apply(model)
            model.eval()
            imgs = sample(model, objective, cfg, n=64, device=device)
            save_grid(imgs, os.path.join(cfg.out_dir, f"sample-{cfg.objective}-{step:06d}.png"))
            model.train()
            ema.restore(model)

        if step % cfg.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow,
                        "optimizer": optimizer.state_dict(), "step": step},
                       os.path.join(cfg.out_dir, f"ckpt-{cfg.objective}-{step:06d}.pt"))

        step += 1

    torch.save({"model": model.state_dict(), "ema": ema.shadow,
                "optimizer": optimizer.state_dict(), "step": step},
               os.path.join(cfg.out_dir, f"ckpt-{cfg.objective}-final.pt"))


def save_grid(imgs, path, nrow=8):
    """[-1,1] -> [0,255] 拼网格保存"""
    from torchvision.utils import make_grid
    grid = make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=nrow, padding=2).clamp(0, 1)
    from PIL import Image
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(path)
    print(f"[sample] saved {path}")


if __name__ == "__main__":
    main()
