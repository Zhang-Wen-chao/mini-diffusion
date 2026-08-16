# 03 · Rectified Flow 与蒸馏：把路径拉直，直到 1 步生成

> 论文：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
> (Liu et al., 2022, arXiv:2209.03003)
> 蒸馏：DMD (arXiv:2310.04157)、DMD2 (arXiv:2405.14867)、MeanFlow 系列 (arXiv:2508.17426 起)
> 目的：理解「直线路径」怎么来（reflow），以及工业界怎么把它压到 1 步（DMD/MeanFlow）。

## 1. 问题：FM 学的路径不直

直线插值路径 $x_t = (1-t)x_0 + tx_1$ 的单条轨迹是直线，但**速度场 $v_\theta$ 学的是
「所有轨迹的平均」**。不同 $(x_0, x_1)$ 对的直线会相交，平均后的轨迹是**弯曲**的：

```
x_0 (噪声) ──┬─ 轨迹A（数据a）
             ├─ 轨迹B（数据b）   → 平均场是弯的 → 欧拉大步长采样有误差
             └─ 轨迹C（数据c）
```

- 若轨迹直：$v(x,t)$ 与 $t$ 无关，欧拉一步 = 精确解，**1 步采样**。
- 若轨迹弯：步数越多越好，几十步才够。

所以两个问题：**(a) 如何让路径更直？（reflow） (b) 学一个 1 步模型？（蒸馏）**

## 2. Rectified Flow 与 Reflow（路径拉直）

**Rectified Flow 目标**：学一个「速度场 $v$ 的积分映射 = 最优传输」的近似，
让轨迹近似直线。**Reflow 过程**（迭代）：

```
1. 训练 v_θ：CFM 目标（同 02 笔记）
2. 用 v_θ 解 ODE，把噪声 x_0 流到 x_1 → 得到配对 (x_0, x_1)
3. 用这些 (x_0, x_1) 配对重新训练一个 v_θ'  （CFM，但配对更"对齐"）
4. 重复 → 轨迹越来越直
```

**直觉**：第一次训练时，$x_0$ 和 $x_1$ 是随机配对的（相交严重 → 弯）。
reflow 后配对是「一条 ODE 轨迹的两端」，自动对齐 → 相交减少 → 更直。
（极限下收敛到 Monge-Kantorovich 最优传输映射，轨迹为直线族。）

**代价**：reflow 要再跑一遍 ODE 生成新配对（$x_0$ 批量 → 采样 → 配对），
相当于多一轮训练数据生成。SD3 等没做 reflow，直接训 FM——说明「直线插值 + 少步数」
已经够用；reflow 是"更直但更贵"的增强。

## 3. 蒸馏：1 步生成的三大流派

目标一致：训练一个网络 $f_\theta(x_T) \to x_0$（或 $g(x_T) \to v$），一次前向出结果。

### 3.1 回归式蒸馏（最简单）

直接让学生学 teacher 的采样轨迹：

$$
\mathcal{L} = \| f_\theta(x) - \mathrm{ODE}_\text{teacher}(x, N\ \text{steps}) \|^2
$$

- 优点：简单稳定。缺点：$N$ 步 teacher 的轨迹随 $N$ 变，蒸馏出来的学生隐式绑定 $N$。
- 代表：Progressive Distillation (Salimans & Ho, 2202.00512)——每轮 teacher 步数减半。

### 3.2 分布匹配蒸馏 DMD / DMD2（概率空间对齐，不用逐点对齐）

**DMD 核心**：学生 $P_\theta$（1 步生成，可写为确定映射 $x = f_\theta(z)$，$z$ 高斯）要逼近 teacher
扩散模型对应的数据分布 $P_\text{data}$。用两种力：

$$
\mathcal{L}_\text{DMD} = \underbrace{\mathbb{E}_{x\sim P_\theta}\big[\left\|s_\text{teacher}(x) - \nabla_x \log P_\theta(x)\right\|^2\big]}_\text{分布匹配项} + \alpha\,\underbrace{\|f_\theta(z) - \hat x_0\|^2}_\text{回归项}
$$

- **分布匹配项**：拉学生分布朝向 teacher 分布（用 teacher 的得分函数 $s_\text{teacher}$，
  teacher 不需要可微——扩散模型得分是现成的）。注意：这**不是**把单个样本逐点对齐，
  而是两个分布的散度下降（推导出来是 score 匹配形式）。
- **回归项**：约束学生输出别漂移太远，稳定训练（$\alpha$ 控制权重）。

**DMD2 改进**（arXiv:2405.14867）：真图样本的分布匹配项（fakereal 双向对抗） +
fake 数据配对的回归项，即**结合 GAN loss**。结果：稳定性和质量大幅提升，
支持 1024×1024 一步生成（SDXL 蒸馏）。

### 3.3 MeanFlow（2025 前沿：时间平均速度场）

**动机**：CM（一致性模型）学「任意 $t \to x_0$ 的映射」；MeanFlow 换个对象——
学**时间平均速度场** $\bar u(x) = \int_0^1 u_t(x)\,dt$（x 处速度对时间的平均）。
平均速度直接给出直线路径方向，1 步欧拉即生成；联合学习瞬时场 $u_t$ 与平均场 $\bar u$
（瞬时场教会平均场"往哪走"）。

- 代表工作：Modular MeanFlow (2508.17426，框架+梯度调制)、
  Understanding/Accelerating/Improving MeanFlow Training (2511.19065，训练动态分析：
  **先有准确瞬时场，才能学好平均场；小时间间隔的平均场要先于大间隔**——与工程
  curriculum 直接相关)、AlphaFlow (2510.20771)、Riemannian MeanFlow (2602.07744，流形版)。
- 与 DMD 的关系：同属"一步蒸馏"大方向；DMD 从分布对齐出发（score/GAN 力），
  MeanFlow 从路径/场出发（更偏连续生成视角，与 consistency 关系更近）。

### 3.4 工业界速览

| 产品/模型 | 技术 | 步数 |
|---|---|---|
| SDXL-Turbo / Lightning | Adversarial + Consistency 蒸馏 | 1-4 |
| LCM / LCM-LoRA | Latent Consistency | 1-4 |
| Wan2.2-Fast | 多阶段少步蒸馏（Wan2.2 发布配套） | 1-4 |
| SD3 家族 | 直接 rectified flow 训练，不蒸馏 | 28-50（原生多步） |

## 4. 关系总图（一条线串起全部）

```
DDPM (2020)        预测噪声，1000 步采样
   │ 时间翻转 + 线性噪声表
Flow Matching      预测速度场，直线条件路径，50~20 步
   │ reflow 拉直（贵）│ 直接训练（SD3/Wan 走这条）
Rectified Flow     更直的轨迹 → 8~4 步
   │ 蒸馏
DMD/DMD2           分布匹配（score + GAN 力）→ 1 步
MeanFlow (2025)    时间平均速度场 → 1 步（前沿，延续 CM 思路）
```

## 5. 与工程实践的关系

- **「跟进和探索前沿技术并落地」**：MeanFlow 系列是 2025 年的热点，一步生成是
  推理成本（IDC 成本）竞争的主战场——DMD2 vs MeanFlow 的取舍是值得关注的细节。
- **「提升训练稳定性」**：蒸馏训练的稳定化（回归项系数、GAN loss 平衡、
  MeanFlow 的 curriculum/梯度调制）就是典型的稳定性工程问题。
- **「训练优化」**：蒸馏对 student 是纯前向训练（无 teacher 采样时），
  数据加载/计算密度与正常训练不同，工具链要适配。

## 6. 一句话总结

> Rectified Flow 用 reflow 把平均轨迹拉直（更少步数）；蒸馏把「多步 ODE」压缩成
> 「一次前向」——DMD/DMD2 从分布对齐发力（score + GAN），MeanFlow 从时间平均速度场
> 发力（2025 前沿）；这条线的核心指标就是「质量 × 步数」的权衡。

**下一步**：04-parallel-reuse.md —— mini-megatron 的并行资产怎么接到 diffusion 训练上（Phase 3 的施工图）。
