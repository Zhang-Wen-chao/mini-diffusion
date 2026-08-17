# 参考手册：论文清单 + 关键对比

> 技术演进图见 [evolution.md](evolution.md)，实现记录见各 Phase 的 experiment-notes。
> 本文保留知识型参考内容（论文清单 / LLM vs Diffusion 对比 / 技术栈）。

## 1. 论文清单

### 1.1 主线（必读）

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

### 1.2 分布式/系统

| 类别 | 论文 | arXiv | 状态 |
|---|---|---|---|
| 并行 | Megatron-LM | 1909.08053 | ✅ mini-megatron 已实现 |
| 并行 | Efficient Large-Scale LM Training (TP) | 2105.09258 | ✅ 同上 |
| 并行 | ZeRO (DeepSpeed) | 1910.02054 | ✅ Phase 4 用工具链跑通 |
| 并行 | Sequence Parallel | 2205.05198 | 长序列（视频）刚需 |
| 并行 | DeepSpeed Ulysses | 2309.14509 | Wan2.2 官方多卡用 FSDP+Ulysses |
| 并行 | Zero-Bubble Pipeline | 2401.10241 | PP 进阶 |
| MoE | Switch Transformer / DeepSeekMoE | 2101.03961 / 2401.06066 | EP 基础 |
| Attention | FlashAttention / 2 / 3 | 2205.14135 / 2307.08691 / 2407.08608 | 原理理解，用现成实现 |
| 框架 | PyTorch FSDP | 2304.11277 | Wan2.2 官方用它 |

### 1.3 扩展阅读

- LCM (Latent Consistency Models, 2310.04378)：LCM-LoRA 微调范式。
- SDXL (2307.01952)、Flux (官网/博客)：工业界主流模型演进。
- Wan2.2 发布说明（GitHub Wan-Video/Wan2.2）：MoE 专家按 SNR 切换、TI2V-5B 压缩。
- Post-Training 相关：RLHF for diffusion、DPO for diffusion。

---

## 2. 关键理解：Diffusion 训练 vs LLM 训练

| 维度 | LLM（GPT） | Diffusion |
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

> 核心结论：LLM 学的并行体系 100% 可迁移；diffusion 的新东西是
> 「目标函数 + 时间条件 + 采样」这套训练范式，以及长序列带来的系统压力。

---

## 3. 技术栈与工具链

| 工具 | 用途 | 本项目用法 |
|---|---|---|
| PyTorch | 底层 | 手写实现（mini 系列传统） |
| accelerate | 单机多卡快速封装 | Phase 3 对比基准之一 |
| DeepSpeed | ZeRO / Ulysses / 现成优化 | Phase 4 跑通 ZeRO-2/3 |
| Megatron-Core | TP/PP 生产实现 | Phase 3 对比基线（已有经验） |
| diffusers | 预训练模型生态 | Phase 1 跑通参考管线 |
| Wan2.2 官方 repo | FSDP + Ulysses 参考 | Phase 4 读代码 + 跑通推理 |
