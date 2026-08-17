# 00 · Diffusion 领域发展综述（论文逐篇摘要）

> 本篇按时间线逐篇摘要扩散生成领域的关键论文：每篇给「要解决的问题 → 核心方法 → 关键结果 → 承上启下的意义」四要素。
> 公式推导见 01（DDPM）、02（Flow Matching）、03（Rectified Flow 与蒸馏）；项目落地见各 Phase 实验记录；论文清单速查见 [reference.md](../reference.md)。

---

## 一、前扩散时代：两种生成范式及其痛点

### VAE — Auto-Encoding Variational Bayes（Kingma & Welling, 2013）

- **问题**：希望像训练分类器一样用似然来训练生成模型，但"逐像素最大化似然"难以直接优化。
- **方法**：引入潜变量 z，编码器 q(z|x) 把图压成分布，解码器从 z 重建 x，用 ELBO（证据下界）训练。
- **结果**：训练稳定，但重建误差 + 潜空间正则导致生成图**偏模糊**。
- **意义**：确立"潜变量 + 下界优化"范式；后续 LDM 直接在 VAE 的 latent 上做扩散，把扩散从像素空间搬进低维空间。

### GAN — Generative Adversarial Nets（Goodfellow et al., 2014）

- **问题**：如何让生成图"看起来真实"而不仅是统计上接近。
- **方法**：生成器与判别器对抗博弈——判别器区分真假，生成器骗过判别器。
- **结果**：生成质量一度领先（风格化、高清晰），一步前向出图。
- **痛点**：训练**不稳定**（模式坍塌、梯度消失/爆炸、超参敏感），且难评估收敛。
- **意义**："对抗/判别器"思路后来被 DMD2、SDXL-Turbo 重新捡起，作为少步蒸馏的最后一脚质量提升。

---

## 二、扩散基础（2020）：把生成建模变成"去噪"

### DDPM — Denoising Diffusion Probabilistic Models（Ho et al., 2020, arXiv:2006.11239）

- **问题**：能否有一个训练稳定、又比 VAE 清晰的生成模型。
- **方法**：前向过程按固定方差表把数据逐步加噪到高斯；反向过程用一个网络学"每步去噪"。损失经 ELBO 展开后，简化成**预测噪声**的回归：`||ε − ε_θ(x_t, t)||²`。
- **结果**：CIFAR-10 上 FID ≈ 3.17，逼近 GAN 而无需对抗训练。
- **意义**：证明"加噪-去噪"框架可以稳定训练出高质量模型；但其采样需要 **1000 步**，推理极慢——后续所有工作（DDIM/flow/蒸馏）都在解决这个问题。

### Score-Based SDE — Score-Based Generative Modeling through SDEs（Song et al., 2020, arXiv:2011.13456）

- **问题**：DDPM 只是"离散加噪"的一种，能否有一个统一理论框架。
- **方法**：把加噪过程看作**随机微分方程（SDE）**的离散化；证明"学打分函数 ∇log p(x)"等价于"学去噪方向"；给出反向 SDE 与等价的概率流 **ODE**，并提出 VE/VP/sub-VP 三种噪声调度。
- **结果**：把 score matching、DDPM、去噪自编码统一到同一 SDE 语言；用 ODE 采样得到更快路径。
- **意义**：为扩散提供了数学根基——**"采样 = 解 ODE"**这个视角，直接催生 DDIM、DPM-Solver 与 Flow Matching。

### DDIM — Denoising Diffusion Implicit Models（Song et al., 2020, arXiv:2010.02502）

- **问题**：DDPM 采样必须 1000 步，太慢。
- **方法**：发现可以推导出一个**非马尔可夫的确定性采样过程**（对应 PF-ODE 的一阶离散化），可以**跳步**采样（如只走 50 步）。
- **结果**：50 步质量与 1000 步接近，采样成本下降一个量级。
- **意义**：证明扩散采样的步数可以大幅压缩；本仓库 Phase 2 的 DDPM 目标即用 DDIM 采样评估。

---

## 三、高分辨率与潜空间（2021-2022）：扩散走出像素空间

### LDM — High-Resolution Image Synthesis with Latent Diffusion Models（Rombach et al., 2021, arXiv:2112.10752）

- **问题**：直接在 512×512 像素上扩散，算力不可承受。
- **方法**：先训一个 VAE 把图像压到低维 latent（约 1/8 分辨率），**在 latent 上做扩散**；条件（文本/类别）通过 cross-attention 注入去噪网络。
- **结果**：高质量文本到图生成，成为 Stable Diffusion 的基础架构。
- **意义**：确立"VAE + latent 扩散 + 条件注入"的现代生成管线；本仓库 Phase 1 用 diffusers 跑通的就是这套管线。

### DPM-Solver — DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Models（Lu et al., 2022, arXiv:2206.00927）

- **问题**：DDIM 的一阶离散化还不够快。
- **方法**：利用扩散 ODE 的**半线性结构**（线性系数只依赖时间 t），做指数积分器形式的高阶求解。
- **结果**：10~20 步达到高质量，成为 SD 生态主流采样器。
- **意义**：把"采样"彻底变成数值 ODE 问题；本仓库未实现（sampler.py 只做了 DDIM 与 Euler），面试如实交代。

---

## 四、Flow 路线（2022-2023）：把路径拉直

### Flow Matching — Flow Matching for Generative Modeling（Lipman et al., 2022, arXiv:2210.02747）

- **问题**：扩散的路径弯曲，且目标（噪声预测）随时间变化剧烈，采样步数难再压缩。
- **方法**：直接定义噪声与数据间的**速度场** `v(x_t, t)`，用**条件流匹配恒等式**把不可解的边缘流匹配化简为"给定配对 (x0, x1) 的条件速度回归"：`||v_θ(x_t,t) − (x1 − x0)||²`。
- **结果**：任意概率路径（包括直线）都能用同一个简单回归目标训练。
- **意义**：给生成建模换了"目标函数"：从预测噪声变成预测速度。SD3/Flux/Wan 全部走这条路线；本仓库 Phase 2 的核心就是它的手写实现。

### Rectified Flow — Flow Straight and Fast: Learning to Generate with Rectified Flow（Liu et al., 2022, arXiv:2209.03003）

- **问题**：FM 中不同 (x0, x1) 配对的直线会**相交**，平均出来的速度场是弯的，少步采样误差仍大。
- **方法**：**reflow 操作**——用当前模型采一批轨迹，把 (x0, x1) 重新配对后重新训练，轨迹逐步拉直；可迭代多轮。
- **结果**：一轮 reflow 即可显著降低少步误差，理论上可逼近直线场。
- **意义**：为"2~4 步出图"提供理论工具；本仓库实现的是直线插值 FM，**未做 reflow**（Phase 2 目标只是 flow vs DDPM 的相对比较）。

---

## 五、一步/少步生成（2022-2025）：把多步 ODE 压缩进一次前向

### Progressive Distillation — Progressive Distillation for Fast Sampling of Diffusion Models（Salimans & Ho, 2022, arXiv:2202.00512）

- **问题**：能否把采样步数压缩到个位数，而不改模型架构。
- **方法**：**2 步合 1 步**——teacher 在两点之间跑 2 步采样，学生学"1 步到达同样终点"；再把学生当 teacher 重复。逐轮把 1024 步减半到 8 步。
- **结果**：8 步采样保持高质量，训练只需简单回归损失。
- **意义**：最早的少步蒸馏路线，**最稳但多轮重蒸成本高**；后续 CM/DMD 都是对"一步生成"更激进的替代。

### Consistency Models — Consistency Models（Song et al., 2023, arXiv:2303.01469）

- **问题**：少步蒸馏是否一定要 teacher 跑采样轨迹。
- **方法**：直接学**自一致性映射** `f(x_t, t) → x_0`：同一条 ODE 轨迹上的任意两个点映射后必须落在同一点。可蒸馏训练（CT，用 teacher 轨迹）或**不依赖 teacher 从零训练（CD）**。
- **结果**：1~2 步采样达到与多步相当的质量，且 CD 摆脱了对 teacher 的依赖。
- **意义**：给出"一步生成"的**约束式**方案；MeanFlow（2025）后来证明 CM 等价于学速度场的**时间平均**。

### DMD — Distribution Matching Distillation（Yin et al., 2023, arXiv:2310.04157）

- **问题**：学生只对齐 teacher 的**单条轨迹终点**（回归），输出分布整体可能偏。
- **方法**：两个损失——**回归项**对齐 teacher 多步 ODE 终点；**分布匹配项**用 score 差（teacher 流形 vs 学生分布）把学生的**输出分布**拉向 teacher 流形。
- **结果**：CIFAR-10 上一步生成 FID 与多步 teacher 差距大幅缩小。
- **意义**："回归 + 分布匹配"双目标成为一步蒸馏的标准配方；本仓库 Phase 5 的回归蒸馏 + DMD 流形项（无对抗简化版）就是这条线。

### DMD2 — One-Step Diffusion with Distribution Matching Distillation（Yin et al., 2024, arXiv:2405.14867）

- **问题**：DMD 质量有上限——分布匹配只对齐低阶统计。
- **方法**：在 DMD 双目标上再加**真图 GAN 对抗项**（判别器区分学生输出与真实图）。
- **结果**：一步生成质量显著提升，逼近多步模型。
- **代价**：GAN 式训练不稳定（博弈、调参敏感）。
- **意义**：一步生成的质量天花板方案；本仓库未做（教学版本选稳定路线），面试主动交代。

### MeanFlow — MeanFlow（2025, arXiv:2508.17426 系列）

- **问题**：CM 与 FM 两条一步生成路线能否统一。
- **方法**：证明 CM 本质上在学**速度场的时间平均**——把多步 ODE 的累积效果等价为一个"平均速度场"，一步生成就是在这个平均场上走一大步。
- **意义**：2025 年前沿的统一直觉，用于理解整个一步生成家族；本仓库仅在演进图与面试 Q&A 中覆盖。

### LCM — Latent Consistency Models（2023, arXiv:2310.04378）

- **方法**：把一致性蒸馏应用到 LDM 的 latent 空间，并给出 **LCM-LoRA** 微调范式：给现成 SD 模型挂 LoRA 蒸馏成 2~4 步。
- **意义**：让少步生成成为"微调可得"的工程方案，工业界快速跟进（SDXL-Turbo、Wan2.2-Fast 同族逻辑）。

---

## 六、架构演进（2021-2024）：去噪网络从卷积到 Transformer

### U-Net（扩散版, 2020-2021）

- **特点**：卷积多分辨率金字塔 + skip 连接；时间步经 embedding 调制 GroupNorm。
- **局限**：局部卷积归纳偏置对大数据扩展性不如 attention；卷积按通道/空间做 TP 切分很别扭。

### DiT — Scalable Diffusion Models with Transformers（Peebles & Xie, 2022, arXiv:2212.09748）

- **方法**：把 U-Net 换成 **patchify + Transformer block**（attention + FFN），时间/类别条件经 **adaLN-Zero** 调制成 6 路 scale/shift/gate。
- **结果**：随模型与算力增长质量持续提升（scaling law），超越 U-Net。
- **意义**：去噪网络与 LLM **同构**——GPT 的 TP/PP/ZeRO 体系可原样复用。本仓库 Phase 3 的自写 DiT 就是迷你版（hidden 384 / depth 6 / patch 4）。

### MMDiT / SD3 — Scaling Rectified Flow Transformers（2024, arXiv:2403.03206）

- **方法**：双流 Transformer——文本与图像 token 各自独立流 + 交叉注意力；训练目标用 **Rectified Flow**。
- **结果**：大规模文生图的 scaling 实证，SD3 家族架构。
- **意义**：印证"FM/RF + Transformer"路线的上限；Flux 亦属同族。

---

## 七、视频与规模化（2024-2025）：容量-成本解耦

### Wan — Open and Advanced Large-Scale Video Generation Models（2025, arXiv:2503.20314）

- **问题**：视频生成的算力与容量需求远超图像。
- **方法**：flow 目标 + 视频 VAE；Wan2.2 引入 **MoE**——按噪声水平（SNR）分段的**双专家完整模型**，推理时按 t 切换并 offload 未用模型；训练侧用 **FSDP + Ulysses 序列并行**。
- **设计动机**：**总容量-激活成本解耦**（27B 总参 / 14B 激活）。
- **意义**：本仓库 Phase 4 的读码对象；对比实验（同激活参数下强行分流反而更差，FID 181 vs 114）从反面验证了该设计动机。

---

## 八、主线总结（面试 30 秒版）

1. **目标函数线**：DDPM 预测噪声 → Score-SDE 统一成打分/ODE 视角 → Flow Matching 把目标换成**速度回归、路径变直** → Rectified Flow 用 reflow 进一步拉直。SD3/Flux/Wan 都选 flow，本质是"直线路径 + 平稳目标"让少步采样误差最小。
2. **采样步数线**：1000 步（DDPM）→ 50 步（DDIM）→ 10~20 步（DPM-Solver）→ 2~8 步（LCM/SDXL-Turbo）→ **1 步（Progressive Distillation → Consistency → DMD/DMD2 → MeanFlow）**——多步 ODE 的"计算结果"被逐步编码进网络权重。
3. **架构线**：U-Net（卷积归纳偏置）→ DiT（Transformer，与 LLM 同构、并行友好）→ MMDiT（多模态双流）→ MoE（容量-成本解耦）。
4. **系统线**：扩散复用了 LLM 的全部并行体系（TP/PP/DP/ZeRO），新增压力是长序列（SP 刚需）与推理多步迭代（蒸馏刚需）。
