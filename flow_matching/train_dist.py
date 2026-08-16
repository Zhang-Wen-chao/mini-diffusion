"""分布式训练入口：DP / TP（DiT 架构，FM 目标）

用法（L20）：
  # 单卡（等价性基准）
  torchrun --nproc_per_node=1 --master_port=29901 train_dist.py --tp 1 --dp 1 --verify
  # TP=2（数学等价性：loss 与单卡一致）
  NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    torchrun --nproc_per_node=2 --master_port=29902 train_dist.py --tp 2 --dp 1 --verify
  # DP=2
  NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    torchrun --nproc_per_node=2 --master_port=29903 train_dist.py --tp 1 --dp 2

等价性验证（--verify）：所有 rank 用相同初始化种子 + 相同数据序列，
逐 step 对比 loss（TP 各 rank loss 相同；与单卡基准 log 对比 <1e-3）。
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .dit import DiT
from .objectives import build_objective
from .parallel.data_parallel import allreduce_grads, scale_tp_untouched_grads
from .parallel.process_groups import init_model_parallel, set_model_parallel

# 分布式下每个 rank 的数据切片：按 dp_rank 切 CIFAR
def make_dist_loader(batch_size, data_dir, dp_rank, dp_size, split="train"):
    from .data import CIFAR10
    from torch.utils.data import DataLoader
    ds = CIFAR10(split, data_dir)
    n = len(ds)
    per = n // dp_size
    start, end = dp_rank * per, (dp_rank + 1) * per if dp_rank < dp_size - 1 else n
    sub = torch.utils.data.Subset(ds, range(start, end))
    return DataLoader(sub, batch_size=batch_size, shuffle=split == "train",
                      num_workers=4, drop_last=True, persistent_workers=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--objective", choices=["ddpm", "flow"], default="flow")
    ap.add_argument("--arch", choices=["unet", "dit"], default="dit")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")
    ap.add_argument("--verify", action="store_true",
                    help="等价性模式：同种子同数据，输出逐 step loss 供对比")
    ap.add_argument("--data-dir", default=TrainConfig.data_dir)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mpu = init_model_parallel(tp_size=args.tp, pp_size=1)
    set_model_parallel(mpu)

    cfg = TrainConfig(objective=args.objective, data_dir=args.data_dir)
    obj = build_objective(cfg)
    torch.manual_seed(0)                      # 所有 rank 相同种子 → 初始化一致

    model = DiT(cfg.model).to(device)
    # TP：广播完整权重（tp 组内 0 号为准）→ 就地切分
    if args.tp > 1:
        src = dist.get_global_rank(mpu["tp_group"], 0)
        for p in model.parameters():
            dist.broadcast(p.data, src=src, group=mpu["tp_group"])
        model.to_tp()
    model.train()

    nparams = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[cfg] arch=DiT tp={args.tp} dp={args.dp} objective={args.objective} "
              f"params={nparams/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 500))

    # 等价性模式：所有 rank 用相同数据（验证 TP/DP 数学等价）
    if args.verify:
        # 构造确定性数据序列（每 rank 相同）
        torch.manual_seed(0)
        xs = [torch.randn(4, 3, 32, 32) for _ in range(args.max_steps)]
    else:
        loader = make_dist_loader(args.batch_size, args.data_dir,
                                  mpu["dp_rank"], mpu["dp_size"])
        it = iter(loader)

    use_amp = args.mixed_precision != "none"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")

    t0 = time.time()
    step = 0
    while step < args.max_steps:
        if args.verify:
            x = xs[step % len(xs)].to(device)
        else:
            try:
                x = next(it).to(device)
            except StopIteration:
                it = iter(loader)
                x = next(it).to(device)

        t = obj.timesteps(x.shape[0], device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = obj.loss(model, x, t)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        allreduce_grads(model, mpu["dp_group"])
        scale_tp_untouched_grads(model, args.tp)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            sps = args.log_every / max(time.time() - t0, 1e-9)
            t0 = time.time()
            print(f"rank{rank} step {step:5d} | loss {loss.item():.6f} | "
                  f"{sps:.1f} step/s")
        step += 1

    torch.cuda.synchronize()
    if rank == 0:
        print(f"[done] final loss {loss.item():.6f} | "
              f"peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
