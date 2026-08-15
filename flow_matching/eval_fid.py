"""评估：FID + 采样步数扫描（flow vs ddpm 少步对比）

用法：
  python3 -m flow_matching.eval_fid \
      --objective flow --ckpt runs/phase2/flow/ckpt-flow-final.pt \
      --steps-list 1 2 4 8 16 50 --n-samples 2048
  python3 -m flow_matching.eval_fid \
      --objective ddpm --ckpt runs/phase2/ddpm/ckpt-ddpm-final.pt \
      --steps-list 5 10 25 50 --n-samples 2048
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import CIFAR10
from .model import UNet
from .objectives import DDPMObjective, FlowObjective
from .sampler import sample_ddim, sample_euler


def inception_features(dl, device, model_name="inception_v3"):
    """用 InceptionV3 pool3 特征（torchvision 预训练，无需额外依赖）"""
    from torchvision import models, transforms
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                transform_input=False)
    model.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(299, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for x in dl:
            x = tf(x).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            if isinstance(out, torch.Tensor):      # eval 模式返回 logits [B,1000]
                f = out
            else:
                f = out.logits
            feats.append(f.float().cpu())
    return torch.cat(feats).numpy()


def fid_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def generate(model, objective, cfg, n, steps, device):
    shape = (n, cfg.model.in_channels, 32, 32)
    if cfg.objective == "ddpm":
        return sample_ddim(model, objective, shape, steps=steps, device=device)
    return sample_euler(model, objective, shape, steps=steps, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps-list", type=int, nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--data-dir", default=TrainConfig.data_dir)
    ap.add_argument("--ref-feats", default=None, help="缓存的真实图特征 npy（可省每次重算）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(objective=args.objective, data_dir=args.data_dir)
    model = UNet(cfg.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["ema"])          # 用 EMA 权重（与训练采样一致）
    model.eval()
    objective = DDPMObjective(cfg) if cfg.objective == "ddpm" else FlowObjective(cfg)
    print(f"[eval] objective={cfg.objective} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    steps_list = args.steps_list or ([5, 10, 25, 50] if cfg.objective == "ddpm"
                                     else [1, 2, 4, 8, 16, 50])

    # 真实图特征（一次计算，缓存）
    if args.ref_feats and os.path.exists(args.ref_feats):
        ref_mu, ref_sigma = fid_stats(np.load(args.ref_feats))
    else:
        dl = DataLoader(CIFAR10("test", args.data_dir), batch_size=128, num_workers=4)
        real = inception_features(dl, device)
        if args.ref_feats:
            np.save(args.ref_feats, real)
        ref_mu, ref_sigma = fid_stats(real)

    print(f"{'steps':>6} | {'FID':>8} | {'mem_peak':>8}")
    results = []
    for steps in steps_list:
        gen = torch.randn(args.n_samples, cfg.model.in_channels, 32, 32)
        imgs = generate(model, objective, cfg, args.n_samples, steps, device)
        gdl = DataLoader(list(imgs.clamp(-1, 1)), batch_size=128, num_workers=0)
        fake = inception_features(gdl, device)
        mu, sigma = fid_stats(fake)
        fid = frechet(mu, sigma, ref_mu, ref_sigma)
        print(f"{steps:>6} | {fid:>8.2f} | {torch.cuda.max_memory_allocated()/1e9:>7.1f}GB")
        results.append((steps, fid))

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"objective={cfg.objective} n={args.n_samples}\n")
            for s, fid in results:
                f.write(f"{s} {fid:.2f}\n")
        print(f"[eval] saved {args.out}")


if __name__ == "__main__":
    main()
