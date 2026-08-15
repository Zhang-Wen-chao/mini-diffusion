# 执行计划（PLAN）

> 更新：2026-08-15。总目标：从 mini-megatron（LLM 并行）到 AIGC（Diffusion）训练，
> 对标 Shopee AIGC 分布式训练优化工程师 JD。原则：**先理论后实现，实现用纯 PyTorch，
> 对比基线验证**。每个 Phase 有明确验收，复用 4×L20 基准方法论。

## 里程碑

```
M0 (当前)  Phase 0 理论完成 → M1: 能手推 DDPM/Flow Matching 目标，讲清三者关系
M1        Phase 1 diffusers 跑通 → M2: 理解完整 pipeline (VAE→TE→UNet→scheduler)
M2        Phase 2 mini flow matching 可训练出图 → M3: 生成质量肉眼可见
M3        Phase 3 分布式+DiT → M4: loss 曲线与单卡一致, MFU 对比
M4        Phase 4 MoE+DeepSpeed (Wan2.2) → M5: 多卡跑通+讲清实现
M5        Phase 5 蒸馏 → M6: 少步生成对比报告
```

## Phase 0 — 理论（当前）

| 任务 | 产出 | 验收 |
|---|---|---|
| 0a DDPM 笔记 | docs/theory/01-ddpm.md | 能手推 ELBO → 简化损失；能写出训练/采样伪代码 |
| 0b Flow Matching 笔记 | docs/theory/02-flow-matching.md | 能推导条件流匹配恒等式；理解高斯路径 |
| 0c Rectified Flow + 蒸馏 | docs/theory/03-rectified-flow-distill.md | 能讲 reflow 直线化 + DMD/MeanFlow 在蒸馏中的位置 |
| 0d 并行复用点 | docs/theory/04-parallel-reuse.md | 列出 mini-megatron 哪些代码直接进 Phase 3 |

## Phase 1 — diffusers 跑通（L20）

| 任务 | 产出 | 验收 |
|---|---|---|
| 环境准备 | 容器装 diffusers/datasets | 训练脚本能起 |
| 小数据训练 | 单卡 SD1.5 微调（小数据集 ~1k 图） | loss 下降，出图有语义 |
| pipeline 拆解笔记 | docs/phase1/pipeline-notes.md | 讲清 VAE/TE/UNet/scheduler 数据流 |

## Phase 2 — mini flow matching（纯 PyTorch）

| 任务 | 产出 | 验收 |
|---|---|---|
| FM 训练循环 | flow_matching/ 单卡实现 | CIFAR-10 训练出图 |
| DDPM 对照 | 同架构双目标 | loss 曲线对比 |
| 测试 | pytest 单测 + L20 训练 | 单测全绿 + 采样有效 |

## Phase 3 — 分布式 + DiT

| 任务 | 产出 | 验收 |
|---|---|---|
| 复用 mini-megatron | TP/PP/DP/AMP 接入 FM 循环 | 多卡 loss == 单卡（1e-3） |
| DiT 架构 | 时间嵌入 + adaLN | 替换 UNet 后质量不降 |
| 对比基线 | accelerate / Megatron-Core | MFU 对比表 |

## Phase 4 — MoE + DeepSpeed（Wan2.2 路线）

| 任务 | 产出 | 验收 |
|---|---|---|
| Wan2.2 读码 | docs/phase4/wan22-notes.md | 讲清双专家 SNR 切换 + FSDP/Ulysses |
| DeepSpeed | ZeRO-3 + Ulysses 长序列 | 多卡跑通，显存/吞吐记录 |
| 迷你 MoE DiT | 双专家扩散 | 与 dense 对照 |

## Phase 5 — 蒸馏

| 任务 | 产出 | 验收 |
|---|---|---|
| DMD 或 MeanFlow 复现 | 小模型 1 步生成 | 质量 vs 步数对比表 |

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| L20 显存/带宽受限（MFU ~14% 天花板） | 只用小模型/小数据做方法论验证 |
| SD1.5 权重下载慢/失败 | HuggingFace + ModelScope 双通道（l20 clash 代理） |
| 视频训练成本过高 | Phase 4 只读代码 + 跑推理，不训练视频模型 |
