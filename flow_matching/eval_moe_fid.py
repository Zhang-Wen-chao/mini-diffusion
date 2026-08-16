"""MoE vs dense FID 对比（用 train_moe_compare 保存的权重）"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_matching.config import ModelConfig, TrainConfig  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.eval_fid import fid_stats, frechet, inception_features  # noqa: E402
from flow_matching.moe_dit import MoEDiT  # noqa: E402
from flow_matching.objectives import FlowObjective  # noqa: E402
from flow_matching.sampler import sample_euler  # noqa: E402


def main():
    cfg = ModelConfig(hidden_size=384, num_heads=6, depth=6,
                      patch_size=4, image_size=32)
    obj = FlowObjective(TrainConfig())
    ref_mu, ref_sigma = fid_stats(np.load("runs/phase2/real-feats.npy"))
    device = "cuda"

    for arch, cls in [("dense", DiT), ("moe", MoEDiT)]:
        m = cls(cfg).to(device)
        sd = torch.load(f"runs/phase4/loss-{arch}.pt")["model"]
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            imgs = sample_euler(m, obj, (1024, 3, 32, 32), steps=50,
                                device=device)
        dl = DataLoader(list(imgs.clamp(-1, 1)), batch_size=128, num_workers=0)
        f = inception_features(dl, device)
        mu, s = fid_stats(f)
        print(f"{arch} 50步 FID: {frechet(mu, s, ref_mu, ref_sigma):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
