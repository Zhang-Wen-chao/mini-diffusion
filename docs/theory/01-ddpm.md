# 01 · DDPM：去噪扩散概率模型

> 论文：Denoising Diffusion Probabilistic Models (Ho et al., 2020, arXiv:2006.11239)
> 配套：Score-Based Generative Modeling through SDE (Song et al., 2020, arXiv:2011.13456)
> 目的：把「前向加噪 + 反向去噪」的目标函数完整推一遍，这是所有后续（Flow Matching/蒸馏）的坐标系。

## 1. 核心思想

生成模型要学 $p_\text{data}(x)$。直接 MLE 难。扩散的做法：

1. **前向**：从数据 $x_0$ 出发，按固定方差表 $\beta_{1:T}$ 逐步加高斯噪声，$T$ 步后几乎变成纯高斯
   $x_T \sim \mathcal{N}(0, I)$（无参数）。
2. **反向**：学一个神经网络 $p_\theta(x_{t-1}|x_t)$ 逐步去噪，从 $x_T$ 采样回 $x_0$。

因为每步加噪都是高斯，**前向可以在任意时刻 $t$ 一步到位**（关键闭式解）。

## 2. 前向过程的闭式解

定义 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$，则：

$$
q(x_t | x_0) = \mathcal{N}\big(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I\big)
$$

重参数化（代码里就是这么写的）：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, I)
$$

**直觉**：$t$ 越大噪声权重越大；$t=T$ 时 $\bar\alpha_T \approx 0$，$x_T \approx \varepsilon$。

## 3. ELBO 推导（为什么目标长那样）

目标是最大化 $\log p_\theta(x_0)$。标准的变分下界（用 Jensen）：

$$
\log p_\theta(x_0) \ge \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]
$$

展开（$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$，$q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1})$），整理成三项：

$$
L = \underbrace{\mathbb{E}_q[-\log p_\theta(x_0|x_1)]}_{\text{重构项}} \;+\;
     \underbrace{\sum_{t=2}^{T} \mathbb{E}_q\left[\mathrm{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)\right]}_{\text{去噪匹配项}} \;+\;
     \underbrace{\mathrm{KL}\big(q(x_T|x_0)\,\|\,p(x_T)\big)}_{\text{先验项，≈0}}
$$

关键点：第 2 项的**真实后验是闭式已知的**（高斯，前面提到 $\beta_t$ 固定），

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde\mu_t(x_t,x_0),\, \tilde\beta_t I)
$$

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t,\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

两个高斯做 KL 有闭式，于是训练 = 让 $p_\theta(x_{t-1}|x_t)$ 的均值逼近 $\tilde\mu_t$。
参数化 $p_\theta$ 的均值（Ho et al. 的核心 trick）：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \varepsilon_\theta(x_t, t)\right)
$$

代入 KL 并忽略与 $\theta$ 无关的系数，得到**简化目标**：

$$
\boxed{\;\mathcal{L}_\text{simple} = \mathbb{E}_{t,\,x_0,\,\varepsilon}\left[\left\|\varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\; t\big)\right\|^2\right]\;}
$$

**即：预测噪声 $\varepsilon$。** 不预测 $x_0$、不预测 $\mu$——这是大量实验选出来的最优参数化。

## 4. 训练 / 采样算法

```
训练：
  for step in ...:
    x0 ~ p_data;  t ~ U{1..T};  ε ~ N(0, I)
    xt = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε
    loss = ||ε - ε_θ(xt, t)||²        # 就是回归损失，无 KL 计算
    backward, step

采样（DDPM，T 步）：
  x_T ~ N(0, I)
  for t = T..1:
    z = 0 if t==1 else N(0, I)
    x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t)) + √β_t z   # 逐步去噪
  return x_0
```

**与 LLM 训练的对比**：LLM 是「预测下一个 token」的分类目标；扩散是「预测噪声」的回归目标。
但训练循环结构完全一样：采样数据 → 前向 → 算损失 → 反向 → 优化器步。这就是为什么
mini-megatron 的并行体系能直接迁移（Phase 3 复用点）。

## 5. DDIM：把采样变成确定性 ODE

DDIM（arXiv:2010.02502）指出采样可以写成**非马尔可夫**的形式：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}\right) + \sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\,\varepsilon_\theta + \sigma_t z
$$

- $\sigma_t = 0$：确定性采样（DDIM），几乎一致的样本，**可以大步长跳着采**（如 1000 → 20 步）
- 括号里第一项是把 $x_t$ 中「预测出的 $x_0$」单独解出来：$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\varepsilon_\theta}{\sqrt{\bar\alpha_t}}$

**直觉**：模型其实隐式学了「当前样本 → 干净样本」的映射，采样的核心就是不断
「去噪 → 沿方向走」。这个「映射」视角是一致性模型/蒸馏的出发点。

## 6. SDE 视角：扩散 = 随机微分方程

Song et al. 证明前向加噪等价于 SDE：

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

（DDPM 对应 VP-SDE：$f = -\tfrac12\beta(t)x$，$g=\sqrt{\beta(t)}$）
反向去噪等价于**反向 SDE**：

$$
dx = \big[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\big]dt + g(t)\,d\bar w
$$

而**打分函数**（score）与噪声预测的关系：

$$
\nabla_x \log p_t(x_t) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}
$$

**意义**：
1. 解释了「为什么预测 $\varepsilon$」——等价于预测得分场（一个分布的方向导数）。
2. 去掉随机项 $g\,d\bar w$ 得到**概率流 ODE**（PF-ODE），确定性、可大步采样、可精确算对数似然。
3. DDIM 就是 PF-ODE 的一个离散化。Flow Matching 则是另一条路（见 02）。

## 7. 工程要点（对齐 JD「训练稳定性」）

| 项 | 实践 | 原因 |
|---|---|---|
| 时间嵌入 | 每个 block 加 $\mathrm{timestep\_embed}(t)$（正弦/余弦） | 模型要区分不同噪声水平 |
| $t$ 采样 | 均匀 $U\{1..T\}$，或加权（见 FM 笔记） | 均匀最简单，加权控各步贡献 |
| EMA | 采样权重用 EMA 版本 | 稳定采样质量 |
| 网络 | UNet（LDM）/ DiT | 见 Phase 3 |
| 损失尺度 | $L_\text{simple}$ 各步等权 | 简化目标不按 $\tilde\beta$ 加权，实践更好 |

## 8. 一句话总结

> DDPM：固定前向加噪 → ELBO 展开 → 两个高斯 KL 有闭式 → 简化成「预测噪声」的回归目标；
> DDIM/SDE 视角把采样升级成确定性 ODE，为少步采样和后续蒸馏铺路。

**下一步**：02-flow-matching.md —— 把「预测噪声」推广成「预测速度场」，得到更一般的框架。
