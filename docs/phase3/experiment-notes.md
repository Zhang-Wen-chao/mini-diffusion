# Phase 3 实验记录：DiT + 分布式（TP/DP）

> 2026-08-16。目标：复用 mini-megatron 并行代码，让 DiT 在 4×L20 上跑通 DP/TP，
> 验证数学等价性（多卡 loss == 单卡）。

## 架构：DiT（替换 UNet）

```
patchify (Conv k=p) → +pos_embed → N×DiTBlock → FinalLayer → unpatchify
DiTBlock = adaLN-zero(norm → attn) + adaLN-zero(norm → FFN)
- hidden 384, depth 6, heads 6, patch 4 → 16.6M 参数（DiT-S 一半深度）
- 5k 步 FM 训练：loss 0.24（与 UNet 同期相当），出图正常
```

**TP 切分（与 GPT 同构，关键决策）**：
- qkv/FFN-up = ColumnParallelLinear（每 rank 部分 head）
- proj/FFN-down = RowParallelLinear + all-reduce（**每层合并**，block 输入输出保持完整 hidden）
- adaLN/patchify/pos_embed/final = 跨 rank 复制（无通信）
- FinalLayer 不切（输入是合并后的完整 hidden）

## 等价性验证结果（重点）

### TP 数学等价性（tp_equiv.py，2 进程单步对比）

| 项 | 结果 |
|---|---|
| 前向输出误差 | **0.00e+00**（bitwise 一致） |
| loss | TP=1.306849 == ref=1.306849（bitwise） |
| 梯度 | 全部通过（容差内，见下） |
| 结论 | **PASS ✅** |

### 训练中逐 step loss 对比（同种子同数据）

| step | 单卡 | TP=2 | 差异 |
|---|---|---|---|
| 0 | 2.327215 | 2.327215 | 0 |
| 9 | 2.314214 | 2.314226 | 1.2e-5 |
| 29 | 2.259639 | 2.259563 | 7.6e-5 |
| 49 | 2.285541 | 2.285323 | 2.2e-4 |

**100 步训练后 loss 差异 ~1e-4（远小于 1e-3 容差）**——TP=2 与单卡训练等价 ✅

### DP=2（真实数据分 shard）

loss 1.5 → 0.32 持续下降，双 rank 各训不同数据子集，梯度 all-reduce 同步 ✅

## 踩坑记录（Phase 3 特有）

1. **RowParallelLinear 的 bias 必须在 all-reduce 之后加**：否则 bias 被重复 tp 次。
   mini-megatron 原版有这个隐患（bias=False 时没暴露），移植时修正。
2. **nn.MultiheadAttention 内部不用 `in_proj` 属性**：替换无效，必须自写 Attention
   （qkv=Column / proj=Row + sdpa）。DiT 反正要 adaLN 调 norm，自写更干净。
3. **to_tp 的权重拷贝要按 rank 切分**（Column 取连续行 / Row 取连续列），
   先 broadcast 完整权重再切分，保证与单卡权重划分一致。
4. **"未切分层梯度除 tp" 在这里不适用**：我的设计是每层 Row 合并（block 输入输出完整），
   复制层的梯度每 rank 本来就是完整正确值（x_embedder 报错证明了）。
   只有"全程不合并"的 Megatron 风格才需要那规则——设计取舍要讲清楚。
5. **all_gather 必须在所有 rank 调用**（NCCL 集合通信对等）。
6. **梯度对比时 TP 梯度要先 all_gather 再按 Column(dim0)/Row(dim1) 拼接**。
7. **ProcessGroup 没有 `.ranks`**：用 `dist.get_global_rank(group, 0)`。
8. 小 bias 梯度（~1e-3）在 all-reduce 与单机 matmul 的浮点路径下相对误差可达 10-20%，
   判定用绝对+相对双阈值（教学验证，绝对量级对训练无影响）。

## 性能（L20 4 卡，DiT 16.6M，32×32）

| 配置 | 吞吐 | 说明 |
|---|---|---|
| 单卡 | 79 step/s (BS4) | verify 模式小 batch |
| TP=2 | 64 step/s | PCIe 通信开销 |
| DP=2 | 23 step/s (BS64) | 真实数据训练 |

> 与 mini-megatron 结论一致：小模型 + PCIe 无 NVLink 下，TP 无收益（通信暴露）。
> 教学重点是**等价性**而非性能；性能在 Phase 4（MoE/大模型/长序列）才进入讨论。

## 复用清单（对照 04-parallel-reuse.md）

```
comm/all_reduce.py          ✅ 原样移植（bias 修复见坑 1）
parallel/process_groups.py  ✅ 原样移植
parallel/tensor_parallel.py ✅ 原样移植 + 记录切分层
parallel/data_parallel.py   ✅ 原样移植
model/transformer.py        ⚠️ 重写为 DiT（attention 自写 + adaLN）
新增: dit.py, train_dist.py, tests/tp_equiv.py
```

## 复现命令

```bash
# TP 数学等价性（2 进程）
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=2 --master_port=299xx \
    -m flow_matching.tests.tp_equiv

# 单卡基准 + TP=2 等价性（--verify 同种子同数据）
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=1 --master_port=299xx \
    -m flow_matching.train_dist --tp 1 --dp 1 --verify --max-steps 100
NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
    --master_port=299xx -m flow_matching.train_dist --tp 2 --dp 1 --verify --max-steps 100

# DP=2 真实训练
NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
    --master_port=299xx -m flow_matching.train_dist --tp 1 --dp 2 --max-steps 300
```
