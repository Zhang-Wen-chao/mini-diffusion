"""DeepSpeed 跑通验证：ZeRO-2/3 训练我们的 DiT（FM 目标）

对比同一训练循环在 PyTorch（Phase 3 已跑）vs DeepSpeed 引擎下的表现：
- 显存（ZeRO-3 切分参数应更低）
- 吞吐

用法（L20）：
  deepspeed --num_gpus=2 train_ds.py --zero 3
  deepspeed --num_gpus=1 train_ds.py --zero 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# deepspeed 0.19.5 与 nvtx>=0.2.8 的 DummyDomain API 不兼容（push_range 签名变化）
# 直接 patch 掉 profiling 调用（deepspeed 不支持 -m，本文件即入口）
try:
    import nvtx
    import nvtx._lib.lib as _nvtx_lib

    _orig = _nvtx_lib.DummyDomain.push_range

    def _patched(self, *args, **kwargs):
        try:
            return _orig(self, *args, **kwargs)
        except TypeError:
            return None

    _nvtx_lib.DummyDomain.push_range = _patched
except (ImportError, AttributeError):
    pass

from flow_matching.config import TrainConfig  # noqa: E402
from flow_matching.data import get_cifar10_loader  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.objectives import FlowObjective  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=-1,
                    help="deepspeed launcher 注入的本地 rank")
    ap.add_argument("--zero", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import deepspeed

    ds_config = {
        "train_batch_size": args.batch_size * int(os.environ.get("WORLD_SIZE", 1)),
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 2e-4}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_min_lr": 0,
                                                     "warmup_max_lr": 2e-4,
                                                     "warmup_num_steps": 100}},
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": args.zero,
                              "offload_optimizer": {"device": "none"}},
        "gradient_clipping": 1.0,
    }
    cfg = TrainConfig()
    torch.manual_seed(0)
    model = DiT(cfg.model)
    obj = FlowObjective(cfg)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, config_params=ds_config)

    dl = get_cifar10_loader(args.batch_size, cfg.data_dir, num_workers=4)
    it = iter(dl)
    rank = int(os.environ.get("RANK", 0))
    t0 = time.time()
    for step in range(args.steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.cuda().to(torch.bfloat16)
        t = torch.rand(args.batch_size, device="cuda").to(torch.bfloat16)
        xt, z = obj.add_noise(x, t)
        v = (x - z).float()
        vp = model_engine(xt, t).float()
        loss = torch.nn.functional.mse_loss(vp, v)
        model_engine.backward(loss)
        model_engine.step()
        if rank == 0 and step % 50 == 0:
            sps = 50 / max(time.time() - t0, 1e-9)
            t0 = time.time()
            print(f"step {step:5d} | loss {loss.item():.4f} | {sps:.1f} step/s "
                  f"| mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    if rank == 0:
        print(f"[done] final loss {loss.item():.6f} | "
              f"peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
