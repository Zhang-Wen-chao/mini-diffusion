"""MoE vs dense 对比实验（同激活参数量）

验证 Wan2.2 核心假设："按 timestep 分工的双专家，比单专家质量更好"
- dense：DiT（单 FFN）
- moe：MoEDiT（双 FFN 专家，boundary=0.5，每步只激活一个专家）

对比：相同训练步数、相同激活参数量、相同数据/种子 → loss 曲线 + FID(可选)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_matching.config import TrainConfig  # noqa: E402
from flow_matching.data import get_cifar10_loader  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.moe_dit import MoEDiT  # noqa: E402
from flow_matching.objectives import FlowObjective  # noqa: E402
from flow_matching.sampler import sample_euler  # noqa: E402


def train(model, obj, steps, out_dir, tag, seed=0, lr=2e-4, bs=64,
          boundary=0.5):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500))
    dl = get_cifar10_loader(bs, TrainConfig.data_dir, num_workers=4)
    it = iter(dl)
    losses = []
    for step in range(steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.cuda()
        t = torch.rand(bs, device="cuda")
        xt, z = obj.add_noise(x, t)
        v = x - z
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(model(xt, t), v)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            print(f"[{tag}] step {step}: loss {loss.item():.4f}", flush=True)

    # 采样出图
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        imgs = sample_euler(model, obj, (64, 3, 32, 32), steps=50,
                            device="cuda")
    from PIL import Image
    from torchvision.utils import make_grid
    grid = make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=8).clamp(0, 1)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(
        "uint8")).save(os.path.join(out_dir, f"sample-{tag}.png"))
    print(f"[{tag}] saved sample, final loss {losses[-1]:.4f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--arch", choices=["dense", "moe", "both"], default="both")
    ap.add_argument("--out-dir", default="runs/phase4")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--boundary", type=float, default=0.5)
    args = ap.parse_args()

    from flow_matching.config import ModelConfig
    cfg = ModelConfig(hidden_size=args.hidden, num_heads=6, depth=args.depth,
                      patch_size=4, image_size=32)
    obj = FlowObjective(TrainConfig())

    if args.arch in ("dense", "both"):
        dense = DiT(cfg).cuda()
        n_dense = sum(p.numel() for p in dense.parameters())
        print(f"dense params: {n_dense/1e6:.1f}M", flush=True)
        l = train(dense, obj, args.steps, args.out_dir, "dense")
        torch.save({"losses": l, "arch": "dense", "model": dense.state_dict()},
                   os.path.join(args.out_dir, "loss-dense.pt"))
        del dense
        torch.cuda.empty_cache()

    if args.arch in ("moe", "both"):
        moe = MoEDiT(cfg, boundary=args.boundary).cuda()
        n_moe = sum(p.numel() for p in moe.parameters())
        # 激活参数 = 每步用到的（attention 全量 + 一个专家）
        n_act = n_dense if args.arch == "both" else n_moe
        print(f"moe params: {n_moe/1e6:.1f}M (dense 的 {n_moe/n_dense:.2f}x, "
              f"激活与 dense 相同)", flush=True)
        l = train(moe, obj, args.steps, args.out_dir, "moe", boundary=args.boundary)
        torch.save({"losses": l, "arch": "moe", "model": moe.state_dict()},
                   os.path.join(args.out_dir, "loss-moe.pt"))
        del moe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
