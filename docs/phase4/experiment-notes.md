# Phase 4 实验记录：迷你 MoE DiT + DeepSpeed

> 2026-08-16。读码（wan22-notes.md）+ 实现 + 对比实验。
> **核心结论（负结果，教学价值最高）**：同激活参数量下，联合训练的分流 FFN MoE
> **不如 dense**——MoE 的价值在于"总容量↑、激活不变"，而不是"同激活更好"。

## 1. 迷你 MoE DiT（Wan2.2 范式迷你版）

```
MoEDiT = patchify → N×MoEDiTBlock → final
MoEDiTBlock = adaLN-zero(norm → attn) + adaLN-zero(norm → 双专家 FFN)
双专家 = expert_hi(高噪声 t>=boundary) + expert_lo(低噪声) —— 按 t 路由（batch 内 mask）
```

- 参数：16.6M(dense) → 23.7M(moe，1.43x，FFN 翻倍)；**激活参数与 dense 相同**
- 与 Wan2.2 差异：Wan 是**双完整模型分开训练**；我们是单模型双 FFN 联合训练

## 2. 对比实验结果（15k 步，FM 目标，CIFAR-10）

### loss 曲线（每千步均值）

| 千步 | 1 | 5 | 10 | 15 |
|---|---|---|---|---|
| dense | 0.405 | 0.208 | 0.195 | **0.191** |
| moe | 0.422 | 0.207 | 0.197 | **0.192** |

loss 几乎持平（早期 moe 略慢 ~1-4%）。

### 采样质量（50 步 Euler，FID ↓）

| | FID |
|---|---|
| dense | **113.97** |
| moe | 180.53（+58%） |

**loss 相同但 FID 明显更差**——这是本次实验最有价值的发现。

### 为什么 MoE 反而差？（分析）

1. **联合训练 + boundary 分流 = 每个专家只看到一半数据**：boundary=0.5 时
   每专家训练样本减半 → 专家欠训练（各自只学了 t 的一半范围）。
2. **共享 attention 适配两个专家**：attention/adaLN 同时服务两个分布不同的 FFN
   输出 → 相互干扰，整体不协调。
3. **对比设计没有兑现 MoE 的优势**：MoE 的收益是"**总参数 27B、每步激活 14B**"
   （Wan2.2），即容量↑而成本不变；我们的对比是"激活参数相同"→ 容量优势不存在。
4. **Wan2.2 的正确姿势**：**两个完整模型分开训练**（各训各的时间段），
   推理按 t 切换 + offload。不联合训练 → 没有相互干扰，且总容量翻倍。

**结论**：MoE 在扩散上的正确用法是 Wan2.2 式的"**按 timestep 分工的独立模型集**"，
目标是**容量-成本解耦**（27B 模型、14B 推理成本），不是"同激活参数内免费提升质量"。
教学实验证明了"强行分流"的代价。

## 3. DeepSpeed 工具链（跑通）

| 配置 | 吞吐 | 峰值显存/卡 | 备注 |
|---|---|---|---|
| ZeRO-2 (2卡) | 31.6 step/s | 1.58 GB | 仅优化器状态切分 |
| ZeRO-3 (2卡) | 14.1 step/s | 1.61 GB | 参数也切分，通信开销 ~2x |

- loss 收敛正常（1.58 → 0.34 @150 步），DiT + FM 目标在 DeepSpeed 引擎下跑通。
- ZeRO-3 的显存优势在小模型（16.6M）上不明显；真实收益在大模型（>1B）场景。

### DeepSpeed 踩坑（0.19.5）

1. **nvtx 不兼容**：deepspeed 0.19.5 调用 `DummyDomain.push_range(message=...)`，
   nvtx>=0.2.8 签名变化 → 必须 monkey-patch `nvtx._lib.lib.DummyDomain.push_range`。
2. **bf16 模式 dtype 管理**：deepspeed bf16 会把参数 cast 成 bf16，但
   **ZeRO-3 的 Linear 包装不做输入 cast**（`torch.addmm` 直接算）→
   输入必须是 bf16。注意点：
   - 模型输入 x/t 手动 `.to(torch.bfloat16)`
   - 我们 `sinusoidal_embedding` 的 freqs 是 fp32（arange 默认）→ 与 bf16 相乘
     类型提升 → 必须 `freqs.to(t.dtype)`
   - norm 层（affine=False）fp32 计算 OK；loss 前 `.float()` 防精度问题
3. **autocast 反而帮倒忙**：deepspeed bf16 下包 autocast，norm 层会保持 fp32
   输入 × bf16 权重 → dtype 冲突。**手动管理 dtype 更可控**。
4. deepspeed launcher 注入 `--local_rank`，argparse 必须接受（旧版约定）。
5. deepspeed 不支持 `-m` 模块方式，需要脚本文件入口。

## 4. 结论与下一步

- Wan2.2 路线核心可迁移点已验证 + 踩坑记录完整
- 后续（可选）：Wan2.2 式**分开训练双模型**实验（需要 2x 训练预算）；
  Ulysses 序列并行已在 wan22-notes.md 读码，DeepSpeed 跑通为长序列打底

## 复现命令

```bash
# MoE vs dense 对比（15k 步 × 2）
python3 -u -m flow_matching.train_moe_compare --steps 15000

# FID 对比（用保存的权重）
python3 -u -m flow_matching.eval_moe_fid

# DeepSpeed ZeRO-2/3（2 卡）
NCCL_SHM_DISABLE=1 deepspeed --num_gpus=2 train_ds_entry.py --zero 3 --steps 150
NCCL_SHM_DISABLE=1 deepspeed --num_gpus=2 train_ds_entry.py --zero 2 --steps 150
```
