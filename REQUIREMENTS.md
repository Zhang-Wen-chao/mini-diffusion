# mini-diffusion — AIGC 分布式训练需求与技术路线

> 阶段：需求记录 + 技术路线规划（2026-08-15）
> 目标：从已完成的 LLM 分布式训练（mini-megatron）走向 AIGC（Diffusion）训练，
> 建立 AIGC 分布式训练方向的完整知识体系。

---

## 1. 背景

已完成 `mini-megatron`（TP + PP + DP + AMP + ZeRO-1，纯 PyTorch ~1000 行），
理解了 LLM 分布式训练四大件：数据并行、张量并行、流水线并行、混合精度。
下一步是 **AIGC 训练**——即 Diffusion 模型的预训练/微调，以及其分布式优化。

本仓库第一阶段产出是**需求文档（本文）**，后续按 Phase 落地实现
（参考 mini-megatron 的"教学验证 + 对比基线"方法论）。

---

## 2. 学习目标拆解

### 2.1 方向 → 学习任务映射

| 方向 | 对应学习任务 |
|---|---|
| AIGC 分布式训练工具链 | 掌握 accelerate / DeepSpeed / Megatron-Core 工具链，理解其取舍 |
| 计算/通信/存储优化 | FlashAttention、算子融合、ZeRO、序列并行（长序列=视频/图像特征）、激活重计算 |
| 训练稳定性 | timestep 采样策略、loss 尺度、SNR 分析、MoE 训练稳定性（Wan2.2 高/低噪声专家） |
| 前沿技术跟进 | Flow Matching → Rectified Flow → 蒸馏（DMD/DMD2/MeanFlow）路线跟踪 |

### 2.2 已有技能 → 待补技能

| 能力 | 已有（mini-megatron 已覆盖） | 差距 | 补课路径 |
|---|---|---|---|
| 数据/流水线/张量并行 | ✅ 手写实现 TP/PP/DP | 专家并行（EP） | Phase 4：Wan2.2 MoE |
| PyTorch / DeepSpeed / Megatron | ✅ PyTorch + Megatron-Core 基线对比 | DeepSpeed（ZeRO/Ulysses） | Phase 4 |
| CUDA / NCCL / cuDNN | ⚠️ 有 NCCL 多卡实操（TP/PP 通信） | CUDA 算子开发 | 可后置 |
| AIGC 预训练、Diffusion（SD/Flux） | ❌ | 全链路 | 本文 Phase 0-3 |
| Transformer 理解 | ✅ 手写 GPT | DiT / MMDiT 变体 | Phase 3 |
| 训练优化/稳定性分析 | ✅ MFU 诊断框架（L20 基准） | diffusion 特有稳定性问题 | Phase 2-3 |

---

## 3. 知识地图：Diffusion/AIGC 训练全景

### 3.1 范式演进（一条线串起来）

```
GAN (对抗) → VAE (似然下界) → DDPM (去噪扩散, 2020)
   → Score-based SDE (统一视角, 2020)
   → Flow Matching / Rectified Flow (最优传输直线路径, 2022)  ← SD3/Flux/Wan 同路线
   → 一致性模型 CM (2023) → 一步蒸馏 DMD (2023) / DMD2 (2024)
   → MeanFlow (2025, 时间平均速度场, CM 与 FM 的统一)        ← 前沿
```

### 3.2 三条主线

- **主线 A — 扩散基础**：DDPM 的噪声预测目标 → DDIM 加速采样 → SDE 统一打分模型视角。
  理解"训练一个网络预测噪声/速度"，以及时间 t 在训练中如何被采样。
- **主线 B — Flow 路线（现代主流）**：
  Flow Matching 把扩散目标改写为"预测速度场"，Rectified Flow 让轨迹直线化（更少步数采样）。
  **SD3（MMDiT）、Flux、Wan2.x 全部走这条路**。这是工程实现的主战场。
- **主线 C — 一步/少步生成（蒸馏）**：
  DMD（分布匹配蒸馏）、DMD2（+GAN loss）、MeanFlow 系列（联合学习瞬时/平均速度场，
  实现 1 步生成）。Wan2.2-Fast、SDXL-Turbo 等产品化加速都来自这条线。

### 3.3 架构演变（生成模型本身）

```
U-Net (LDM/SD1.x) → DiT (纯 Transformer) → MMDiT (SD3, 双流合并) 
   → Flux (SD3 改进) → Wan2.2 (Diffusion + MoE: 高噪声专家/低噪声专家)
```

- 前置模块：VAE（图像/视频压缩到 latent，Wan2.2-VAE 压缩率 16×16×4=64×）、
  Text Encoder（CLIP / T5 / UMT5-XXL）。
- 视频特有：时间维度（帧间因果/3D attention）、长序列 → 序列并行刚需。

---

## 4. 论文清单

### 4.1 必读（主线）

| 类别 | 论文 | arXiv | 一句话要点 |
|---|---|---|---|
| 扩散基础 | Denoising Diffusion Probabilistic Models (DDPM) | 2006.11239 | 噪声预测训练 + 反向去噪采样 |
| 扩散基础 | Score-Based Generative Modeling through SDE | 2011.13456 | 打分函数统一扩散/噪声/ODE 视角 |
| 采样 | Denoising Diffusion Implicit Models (DDIM) | 2010.02502 | 确定性采样，步数大减 |
| 采样 | DPM-Solver | 2206.00927 | 基于 ODE 指数积分的快速采样 |
| Latent | High-Resolution Image Synthesis with LDM | 2112.10752 | VAE latent 上做扩散（SD 基石） |
| Flow | Flow Matching for Generative Modeling | 2210.02747 | 条件流匹配，任意分布间直线插值 |
| Flow | Flow Straight and Fast (Rectified Flow) | 2209.03003 | 重流让轨迹直线化，少步采样 |
| 一致性 | Consistency Models | 2303.01469 | 学"任意 t → x0"映射，1 步生成 |
| 架构 | Scalable Diffusion Models (DiT) | 2212.09748 | Transformer 取代 U-Net |
| 架构 | Scaling Rectified Flow Transformers (SD3) | 2403.03206 | MMDiT + rectified flow 大规模实证 |
| 蒸馏 | Distribution Matching Distillation (DMD) | 2310.04157 | 用分布匹配把扩散蒸馏到 1 步 |
| 蒸馏 | One-Step Diffusion with DMD2 | 2405.14867 | DMD + 真图对抗损失，质量/稳定性提升 |
| 蒸馏 | MeanFlow 系列（Modular MeanFlow 等） | 2508.17426 等 | 时间平均速度场，1 步生成，2025 前沿 |
| 视频 | Wan: Open and Advanced Large-Scale Video Gen Models | 2503.20314 | Wan2.1/2.2：flow + MoE + 视频 VAE |

### 4.2 分布式/系统（复用 + 新学）

| 类别 | 论文 | arXiv | 状态 |
|---|---|---|---|
| 并行 | Megatron-LM | 1909.08053 | ✅ mini-megatron 已实现 |
| 并行 | Efficient Large-Scale LM Training (TP) | 2105.09258 | ✅ 同上 |
| 并行 | ZeRO (DeepSpeed) | 1910.02054 | ⚠️ 已理解 ZeRO-1，Phase 4 用工具链 |
| 并行 | Sequence Parallel | 2205.05198 | ❌ 长序列（视频）刚需 |
| 并行 | DeepSpeed Ulysses | 2309.14509 | ❌ Wan2.2 官方多卡用 FSDP+Ulysses |
| 并行 | Zero-Bubble Pipeline | 2401.10241 | ❌ PP 进阶 |
| MoE | Switch Transformer / DeepSeekMoE | 2101.03961 / 2401.06066 | ❌ EP 基础 |
| Attention | FlashAttention / 2 / 3 | 2205.14135 / 2307.08691 / 2407.08608 | ⚠️ 原理理解，用现成实现 |
| 框架 | PyTorch FSDP | 2304.11277 | ❌ Wan2.2 官方用它 |

### 4.3 扩展阅读

- LCM (Latent Consistency Models, 2310.04378)：LCM-LoRA 微调范式。
- SDXL (2307.01952)、Flux (官网/博客)：工业界主流模型演进。
- Wan2.2 发布说明（GitHub Wan-Video/Wan2.2）：MoE 专家按 SNR 切换、TI2V-5B 压缩。
- "Post-Training" 相关：RLHF for diffusion、DPO for diffusion（可后置）。

---

## 5. 关键理解：Diffusion 训练 vs LLM 训练

| 维度 | LLM（GPT，已掌握） | Diffusion（待学） |
|---|---|---|
| 目标函数 | 下一 token 交叉熵 | 噪声/速度回归（MSE） |
| 训练样本构造 | 原始序列 | 随机采样 t，加噪/插值得到 (x_t, t) 对 |
| 时间维 | 无 | t 是关键条件，采样策略影响稳定性（logit-normal、SNR 加权） |
| 条件输入 | token | text embedding + 可选图/音频 |
| 前向成本 | 1 次 | 1 次（训练时）；采样需 N 步迭代（推理贵） |
| 序列长度 | 几 K token | 图像/视频 latent token 可到 10K+（SP/重计算刚需） |
| 激活显存 | 大 | 更大（长序列），activation checkpoint 更关键 |
| 数据 | 文本 | 图文/视频 pair，清洗与标注管线不同 |
| 并行手段 | TP/PP/DP/AMP/ZeRO | 完全复用，另加 EP（MoE）、SP（长序列） |

> 核心结论：**mini-megatron 学的并行体系 100% 可迁移**，diffusion 的新东西是
> 「目标函数 + 时间条件 + 采样」这套训练范式，以及长序列带来的系统压力。

---

## 6. 技术栈与工具链

| 工具 | 用途 | 本项目用法 |
|---|---|---|
| PyTorch | 底层 | 手写实现（mini 系列传统） |
| accelerate | 单机多卡快速封装 | Phase 3 对比基准之一 |
| DeepSpeed | ZeRO / Ulysses / 现成优化 | Phase 4 工具链 |
| Megatron-Core | TP/PP 生产实现 | Phase 3 对比基线（已有经验） |
| diffusers | 预训练模型生态 | Phase 1 跑通参考管线 |
| Wan2.2 官方 repo | FSDP + Ulysses 参考 | Phase 4 读代码 + 跑通推理 |
| L20 4 卡 + NGC PyTorch 容器 | 实验环境 | 全部实验 |

硬件约束（沿用 mini-megatron 结论）：L20 GDDR6 带宽 864 GB/s，单卡 MFU 天花板 ~14%，
PCIe 无 NVLink。做"正确性/方法论"验证，不做规模。

---

## 7. 学习路线（Phase 计划）

### Phase 0 — 理论（论文精读，产出笔记）
- DDPM → Flow Matching → Rectified Flow 数学推导笔记（docs/）
- 输出：`docs/theory/` 三篇笔记，手推公式

### Phase 1 — 用现成工具跑通
- diffusers + 小数据集（如 cifar/celeba 子集）跑通 SD 训练
- 理解完整 pipeline：VAE → text encoder → UNet/DiT → scheduler
- 输出：训练/推理跑通记录 + 关键组件拆解笔记

### Phase 2 — 手写 mini flow matching（本仓库核心实现）
- 纯 PyTorch 实现 Flow Matching 训练（~500-800 行），参考 mini-megatron 风格
- 数据集：CIFAR-10（或简化 toy）
- 实现：条件流匹配目标、rectified flow 的 reflow 过程、DDPM→FM 双目标对比
- 验证：生成质量（FID 可选）/ loss 曲线；CPU 单测 + L20 训练

### Phase 3 — 分布式 + DiT
- 把 mini-megatron 的 TP/PP/DP/AMP 接到 diffusion 训练循环
- 架构升级：U-Net → DiT（含时间嵌入、无分类器条件）
- 对比：accelerate / Megatron-Core 同配置的 loss 曲线 + MFU（沿用 L20 基准方法论）

### Phase 4 — MoE + 前沿工具链
- Wan2.2 架构复刻（迷你版）：双专家（高/低噪声 SNR 切换）+ 视频 VAE 理解
- 用 DeepSpeed（ZeRO-3 / Ulysses）跑通长序列训练，理解 SP
- 读 Wan2.2 官方多卡推理/训练代码（FSDP + Ulysses）

### Phase 5 — 蒸馏（可选进阶，前沿跟踪）
- 复现 DMD 或 MeanFlow 一步蒸馏（小模型 + 小数据）
- 输出：与多步基线（Phase 2）的 quality/step 对比

---

## 8. 验收标准

- Phase 2：训练 loss 正常下降，采样生成肉眼可见有效图像；CPU 单测覆盖核心逻辑
- Phase 3：分布式结果与单卡 loss 曲线一致（差值 < 1e-3），MFU 接近同配置基线
- Phase 4：跑通 Wan2.2 推理（多卡），理解其 MoE/SP/FSDP 实现并能讲清楚
- 总结输出：能完整讲清「从 LLM 到 diffusion 训练」差异 + 并行体系复用点

---

## 9. 参考资源

- 仓库：github.com/Wan-Video/Wan2.2、huggingface.co/diffusers、github.com/microsoft/DeepSpeed
- 论文入口：以上 arXiv 编号
- 已有资产：mini-megatron（并行实现可复用）、L20 基准方法论（handoff 第 14-15 节）
