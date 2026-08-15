# 02 · Flow Matching：从「预测噪声」到「预测速度场」

> 论文：Flow Matching for Generative Modeling (Lipman et al., 2022, arXiv:2210.02747)
> 工程落地：SD3 (arXiv:2403.03206)、Flux、Wan2.x —— 现代 AIGC 的主流训练目标。
> 目的：理解 FM 目标为何是「回归速度」，以及那个著名的「无条件 = 条件期望」恒等式。

## 1. 动机：扩散只是「概率路径」的一种

DDPM 定义了一条从数据到高斯的路径（$\bar\alpha_t$ 这条）。Flow Matching 问：
**能不能任意设计「分布如何从噪声变成数据」的路径，并直接学路径对应的速度场？**

好处：路径可以选「直线」——直线路径用欧拉法几步就能采好（甚至 1 步），
这是 Rectified Flow / 少步采样的根源。

## 2. 数学设定

定义一个**概率路径** $p_t(x)$，$t \in [0,1]$：$p_0 = $ 高斯噪声，$p_1 = $ 数据分布。
再定义一个**速度场** $u_t(x)$，用它生成一个 ODE（连续归一化流 CNF）：

$$
\frac{dx}{dt} = u_t(x),\qquad x(0) \sim p_0
$$

若速度场正确，$x(t) \sim p_t$（即 $p_t$ 是概率分布的"流"）。分布随时间的变化由
**连续性方程**（概率守恒）约束：

$$
\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (p_t\, u_t)
$$

Flow Matching 目标：直接用回归学 $u_t$：

$$
\boxed{\;\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t\sim U[0,1],\; x\sim p_t}\left[\left\|v_\theta(x,t) - u_t(x)\right\|^2\right]\;}
$$

## 3. 关键困难 + 核心定理（CFM 恒等式）

**困难**：$u_t(x)$ 是对所有数据的平均速度场，边缘分布 $p_t$ 和平均速度 $u_t$ 都**没有闭式**（除非已知数据分布）。

**解法**：定义**条件**速度场 $u_t(x|x_1)$（从单个样本 $x_1$ 出发的路径），且条件路径
$p_t(x|x_1)$ 有闭式（高斯）。则：

$$
\mathcal{L}_\text{FM}(\theta) = \mathcal{L}_\text{CFM}(\theta)
$$

$$
\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,x_1\sim q,\,x\sim p_t(\cdot|x_1)}\left[\left\|v_\theta(x,t) - u_t(x|x_1)\right\|^2\right]
$$

**证明思路（30 秒版）**：定义
$u_t(x) = \int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1$，
可得 $u_t(x) = \mathbb{E}_{x_1|x_t=x}[u_t(x|x_1)]$（条件期望），即平均速度是条件速度的期望。
展开 $\|v_\theta - u_t\|^2$ 对 $p_t(x)= \int p_t(x|x_1)q(x_1)dx_1$ 积分，
交叉项用条件期望性质抵消，得到两目标相差一个与 $\theta$ 无关的常数。
（详见 Lipman 论文 Proposition 1-3；面试讲「条件期望 + 交叉项消掉」就够。）

**直觉**：无条件目标等价于——随机挑一个样本 $x_1$、随机挑时间 $t$、在 $x_1$ 的条件路径上采样 $x_t$，回归「条件速度」。不需要知道任何全局量。

## 4. 高斯条件路径 + 直线插值（实际用的形式）

选择线性插值路径（SD3 / Rectified Flow / Wan 全用这个）：

$$
x_t = (1-t)\,x_0 + t\,x_1,\qquad x_0 \sim \mathcal{N}(0,I)\ \text{(噪声)},\; x_1 \sim q\ \text{(数据)}
$$

- $t=0$：纯噪声；$t=1$：纯数据。**路径是直线**，条件速度是常数：

$$
u_t(x|x_1) = \frac{d x_t}{dt} = x_1 - x_0
$$

训练损失（代码就是这一行）：

$$
\boxed{\;\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\,x_0,\,x_1}\left[\left\|v_\theta\big((1-t)x_0 + t x_1,\; t\big) - (x_1 - x_0)\right\|^2\right]\;}
$$

**即：预测「数据 − 噪声」= 速度**。采样时：

```
x_T ~ N(0, I)
for t = T-1..0:                 # 欧拉法解 ODE dx/dt = v_θ
    x_t = x_{t+1} - v_θ(x_{t+1}, t+1) * Δt
```

步数任意（50 步 → 20 步 → 4 步 → 1 步），路径越直，少步误差越小。

## 5. 与 DDPM 的关系（时间翻转 + 变量代换）

约定差：DDPM 的 $t$ 从数据往噪声走，FM 从噪声往数据走；把 DDPM 时间翻转
$\tau = 1-t$ 且用 $\bar\alpha_\tau = (1-\tau)^2$（线性噪声表）时：

| 视角 | 预测对象 | 与噪声/数据的关系 |
|---|---|---|
| $\varepsilon$-prediction | 噪声 $\varepsilon$ | $\varepsilon \propto x_t - \sqrt{\bar\alpha}\,x_0$ |
| $v$-prediction | 速度 $v$ | $v = x_1 - x_0$（linear schedule） |
| $x_0$-prediction | 干净样本 | $x_0 = x_t - \sqrt{1-\bar\alpha}\,\varepsilon$ |

三者只是同一条概率路径的不同参数化，**速度预测在这条路径上是"最平"的目标**
（$v$ 对 $t$ 是常数 → 回归目标更稳 → 少步采样误差小）。这是 Flow 路线赢在采样效率的核心。

## 6. 时间采样与训练稳定性（对齐 JD）

- 均匀 $t \sim U[0,1]$：基线。
- **Logit-normal 采样**（SD3 用）：$t \sim \sigma(\mathcal{N}(\mu,\sigma))$，
  把采样权重偏向中间噪声水平（那里的信噪比最"难"、最需要学）。SD3 论文实测质量提升。
- **SNR 加权**：各 $t$ 的损失乘以与信噪比相关的权重，控制每步贡献。
- 实践共识：flow 目标比 $\varepsilon$-prediction 更容易训稳定（损失尺度、梯度尺度更均匀）。

## 7. 工程要点（SD3 / Wan 的做法）

| 项 | SD3 / Wan 实践 |
|---|---|
| 路径 | 直线插值 + $v$-prediction |
| 时间嵌入 | 转成 SNR 的 log 值做正弦嵌入（$\log(\text{SNR})$） |
| 时间采样 | logit-normal / 均匀 |
| 网络 | MMDiT（双流，文本+图像各自 transformer 后合并） |
| latent | VAE 压缩后空间上做 flow（Wan2.2-VAE 16×16×4） |
| 文本编码 | T5 / UMT5-XXL（Wan 用 umt5-xxl） |

## 8. 一句话总结

> Flow Matching：把扩散的「预测噪声」推广为「预测任意概率路径的速度场」；
> 核心定理「条件流匹配」把不可解的无条件目标变成可采样的条件回归；
> 选直线路径 + 速度预测 → 目标最平、采样可少步——这就是 SD3/Flux/Wan 都用它的原因。

**下一步**：03-rectified-flow-distill.md —— 路径如何进一步"拉直" + 一步生成（蒸馏）。
