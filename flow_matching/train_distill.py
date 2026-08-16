"""Phase 5: 蒸馏——一步生成

Teacher: Phase 2 的 flow 模型（60k 步，50 步 Euler FID 54.4）
Student: 新 DiT g_θ(z) = x̂（一步从噪声到图像，t 固定 0）

方法（对应 03-rectified-flow-distill.md）：
1. 回归蒸馏: L_reg = ||g_θ(z) - ODE_teacher(z, 25步)||²  —— DMD 的回归项
2. 流形一致性（DMD 分布匹配的简化实现）:
   L_dm = ||x̂ - [x̂_t + (1-t)·v_teacher(x̂_t, t)]||²
   —— x̂ 加噪后在 teacher 场下一步重建应回到自身（x̂ 落在 teacher 流形上）

工程优化：teacher ODE 目标缓存（数据集蒸馏风格），每 N 步刷新。
验收: 1 步 FID vs teacher 1 步 Euler（860）
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_matching.config import ModelConfig, TrainConfig  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.objectives import FlowObjective  # noqa: E402


@torch.no_grad()
def teacher_sample(teacher, obj, z, steps, device="cuda"):
    """teacher ODE 采样（Euler midpoint）"""
    dt = 1.0 / steps
    x = z
    for i in range(steps):
        t = torch.full((z.shape[0],), (i + 0.5) * dt, device=device)
        x = x + dt * teacher(x, t)
    return x


class DistillTargetCache:
    """缓存 (z, x_teacher) 对，每 refresh_every 步重新生成（数据集蒸馏风格）"""

    def __init__(self, teacher, obj, n=512, ode_steps=25, refresh_every=200,
                 device="cuda"):
        self.teacher = teacher
        self.obj = obj
        self.n = n
        self.ode_steps = ode_steps
        self.refresh_every = refresh_every
        self.device = device
        self.refresh()

    def refresh(self):
        z = torch.randn(self.n, 3, 32, 32, device=self.device)
        self.z = z
        self.x_t = teacher_sample(self.teacher, self.obj, z, self.ode_steps,
                                  self.device)
        print(f"[cache] refreshed {self.n} teacher targets", flush=True)

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.n, (batch_size,), device=self.device)
        return self.z[idx], self.x_t[idx]

    def maybe_refresh(self, step):
        if step % self.refresh_every == 0 and step > 0:
            self.refresh()


def manifold_loss(teacher, obj, x_hat, t_val=0.8):
    """x̂ 加噪后在 teacher 场下一步重建应回到自身（teacher 流形约束）"""
    t = torch.full((x_hat.shape[0],), t_val, device=x_hat.device)
    with torch.no_grad():
        x_t, z = obj.add_noise(x_hat, t)
        v_teacher = teacher(x_t, t)
        x_recon = x_t + (1 - t[:, None, None, None]) * v_teacher
    return torch.nn.functional.mse_loss(x_hat, x_recon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt", default="runs/phase2/flow/ckpt-flow-final.pt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--teacher-ode-steps", type=int, default=25)
    ap.add_argument("--use-dmd", action="store_true", help="加流形一致性项")
    ap.add_argument("--dmd-lambda", type=float, default=1.0)
    ap.add_argument("--out-dir", default="runs/phase5")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    cfg = ModelConfig(hidden_size=384, num_heads=6, depth=6,
                      patch_size=4, image_size=32)
    obj = FlowObjective(TrainConfig())

    from flow_matching.model import UNet
    teacher = UNet(cfg).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[teacher] params={sum(p.numel() for p in teacher.parameters())/1e6:.1f}M",
          flush=True)

    student = DiT(cfg).to(device)
    print(f"[student] params={sum(p.numel() for p in student.parameters())/1e6:.1f}M",
          flush=True)

    cache = DistillTargetCache(teacher, obj, n=512, ode_steps=args.teacher_ode_steps)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000))

    os.makedirs(args.out_dir, exist_ok=True)
    for step in range(args.steps):
        z, x_teacher = cache.sample_batch(args.batch_size)
        x_student = student(z, torch.zeros(args.batch_size, device=device))
        loss = torch.nn.functional.mse_loss(x_student, x_teacher)
        if args.use_dmd:
            loss = loss + args.dmd_lambda * manifold_loss(teacher, obj, x_student)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        cache.maybe_refresh(step)
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.4f}", flush=True)

    tag = "reg" if not args.use_dmd else "dmd"
    torch.save({"model": student.state_dict(), "steps": args.steps},
               os.path.join(args.out_dir, f"student-{tag}.pt"))
    print(f"[saved] {args.out_dir}/student-{tag}.pt", flush=True)


if __name__ == "__main__":
    main()
