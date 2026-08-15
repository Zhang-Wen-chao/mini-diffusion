"""用指定权重(EMA 或 raw)评估 FID 对比"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import CIFAR10
from .eval_fid import fid_stats, frechet, inception_features
from .model import UNet
from .objectives import DDPMObjective, FlowObjective
from .sampler import sample_ddim, sample_euler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["ddpm", "flow"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["model", "ema"], default="ema")
    ap.add_argument("--steps-list", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--ref-feats", required=True)
    ap.add_argument("--data-dir", default=TrainConfig.data_dir)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(objective=args.objective, data_dir=args.data_dir)
    obj = DDPMObjective(cfg) if cfg.objective == "ddpm" else FlowObjective(cfg)
    model = UNet(cfg.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt[args.weights])
    model.eval()

    ref_mu, ref_sigma = fid_stats(np.load(args.ref_feats))
    print(f"{'steps':>6} | {'FID':>8}")
    for steps in args.steps_list:
        if cfg.objective == "ddpm":
            imgs = sample_ddim(model, obj, (args.n_samples, 3, 32, 32),
                               steps=steps, device=device)
        else:
            imgs = sample_euler(model, obj, (args.n_samples, 3, 32, 32),
                                steps=steps, device=device)
        dl = DataLoader(list(imgs.clamp(-1, 1)), batch_size=128, num_workers=0)
        f = inception_features(dl, device)
        mu, sigma = fid_stats(f)
        print(f"{steps:>6} | {frechet(mu, sigma, ref_mu, ref_sigma):>8.2f}")


if __name__ == "__main__":
    main()
