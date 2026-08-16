"""蒸馏评估：student 1 步 FID vs teacher 多步 FID 对比表"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_matching.config import ModelConfig, TrainConfig  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.eval_fid import fid_stats, frechet, inception_features  # noqa: E402
from flow_matching.model import UNet  # noqa: E402
from flow_matching.objectives import FlowObjective  # noqa: E402
from flow_matching.sampler import sample_euler  # noqa: E402


def fid_of(model, obj, n, steps, device="cuda", z=None):
    if z is None:
        z = torch.randn(n, 3, 32, 32, device=device)
    if steps == 1 and not isinstance(model, UNet):
        imgs = model(z, torch.zeros(n, device=device))
    else:
        imgs = sample_euler(model, obj, (n, 3, 32, 32), steps=steps, device=device)
    dl = DataLoader(list(imgs.clamp(-1, 1)), batch_size=128, num_workers=0)
    f = inception_features(dl, device)
    mu, s = fid_stats(f)
    return frechet(mu, s, *ref_stats()), f


def ref_stats():
    ref = np.load("runs/phase2/real-feats.npy")
    return fid_stats(ref)


def main():
    device = "cuda"
    cfg = ModelConfig(hidden_size=384, num_heads=6, depth=6,
                      patch_size=4, image_size=32)
    obj = FlowObjective(TrainConfig())
    n = 2048
    z = torch.randn(n, 3, 32, 32, device=device)

    # teacher 多步参考
    teacher = UNet(cfg).to(device)
    ckpt = torch.load("runs/phase2/flow/ckpt-flow-final.pt", map_location=device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for steps in [1, 8, 50]:
        f, _ = fid_of(teacher, obj, n, steps, z=z)
        print(f"teacher {steps:3d}步: FID {f:.2f}", flush=True)

    # student 一步
    for tag in ["reg", "dmd"]:
        p = f"runs/phase5/student-{tag}.pt"
        if not os.path.exists(p):
            continue
        student = DiT(cfg).to(device)
        student.load_state_dict(torch.load(p, map_location=device)["model"])
        student.eval()
        f, _ = fid_of(student, obj, n, 1, z=z)
        print(f"student({tag}) 1步: FID {f:.2f}", flush=True)


if __name__ == "__main__":
    main()
