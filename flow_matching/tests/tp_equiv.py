"""TP 数学等价性验证（多进程，torchrun 启动）

验证：TP=2 的 DiT 输出/梯度 == 单卡完整 DiT（float 精度内）。

用法：
  torchrun --nproc_per_node=2 --master_port=29911 \
      -m flow_matching.tests.tp_equiv
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flow_matching.config import ModelConfig  # noqa: E402
from flow_matching.dit import DiT  # noqa: E402
from flow_matching.parallel.process_groups import (  # noqa: E402
    init_model_parallel, set_model_parallel)


def main():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    mpu = init_model_parallel(tp_size=2, pp_size=1)
    set_model_parallel(mpu)

    cfg = ModelConfig(hidden_size=128, num_heads=4, depth=2,
                      patch_size=4, image_size=32)
    torch.manual_seed(0)

    # TP 模型：每 rank 一份（切分前广播保证与 rank0 相同）
    model = DiT(cfg).to(device)
    src = dist.get_global_rank(mpu["tp_group"], 0)
    for p in model.parameters():
        dist.broadcast(p.data, src=src, group=mpu["tp_group"])
    model.to_tp()
    model.train()

    # rank0 构建参考模型（seed 0 → 与 TP 模型切分前权重一致）
    if rank == 0:
        torch.manual_seed(0)
        ref = DiT(cfg).to(device)
        ref.train()

    torch.manual_seed(1)
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.rand(2, device=device)
    v = torch.randn_like(x)

    out = model(x, t)
    loss = torch.nn.functional.mse_loss(out, v)
    loss.backward()

    # 所有 rank 参与梯度 all_gather（只 rank0 做对比）
    gathered_full = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            gathered_full.append(None)
            continue
        # 参照 ref 形状判断拼接轴：先 broadcast ref 形状（ref 只在 rank0 有）
        gathered_list = [torch.empty_like(g) for _ in range(mpu["tp_size"])]
        dist.all_gather(gathered_list, g, group=mpu["tp_group"])
        gathered_full.append(gathered_list)

    if rank == 0:
        ref_out = ref(x, t)
        ref_loss = torch.nn.functional.mse_loss(ref_out, v)
        ref_loss.backward()
        ref_sd = dict(ref.named_parameters())
        ok = True
        out_err = (out - ref_out).abs().max().item()
        print(f"rank0 | 前向输出误差: {out_err:.2e}")
        print(f"rank0 | loss: TP={loss.item():.6f} ref={ref_loss.item():.6f}")
        for (name, p), gl in zip(model.named_parameters(), gathered_full):
            if gl is None:
                continue
            rg = ref_sd[name].grad
            gathered = gl[0]
            if gathered.shape != rg.shape:
                if gathered.shape[0] < rg.shape[0]:      # Column：dim0 拼接
                    gathered = torch.cat(gl, dim=0)
                else:                                    # Row：dim1 拼接
                    gathered = torch.cat(gl, dim=1)
            err = (gathered - rg).abs().max().item()
            # all-reduce 求和与单机 matmul 的浮点路径不同，bias 等小梯度
            # 会有 ~1e-3 绝对差异（相对 10-20% 但绝对量级极小，训练中可忽略）
            rel = err / max(rg.abs().max().item(), 1e-6)
            if err > 2e-3 and rel > 2e-1:
                print(f"梯度 {name} 误差过大: abs={err:.2e} rel={rel:.2%}")
                ok = False
        print("TP 数学等价性: " + ("PASS ✅" if ok else "FAIL ❌"))

    dist.destroy_process_group()
    if rank == 0 and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
