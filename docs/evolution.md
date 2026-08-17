# 技术演进路线

两条线：**方法演进**（Diffusion 领域从 DDPM 到一步生成的路线，标注每个节点
解决什么问题）和**项目演进**（本仓库 Phase 0-5 的落地路径与量化结果）。

---

## 1. 方法演进（领域视角）

```mermaid
flowchart TD
    subgraph 生成范式
        GAN[GAN 2014<br/>对抗训练, 不稳定]
        VAE[VAE 2013<br/>似然下界, 图像模糊]
    end

    subgraph 扩散基础 2020-2021
        DDPM[DDPM 2020<br/>去噪扩散, 1000 步采样<br/>核心: ELBO→简化损失 预测噪声]
        SDE[Score-based SDE 2020<br/>统一扩散/打分/ODE 视角]
        DDIM[DDIM 2020<br/>确定性采样 ODE 离散化]
        LDM[LDM 2021<br/>VAE latent 上扩散<br/>→ Stable Diffusion 家族]
    end

    subgraph Flow 路线 2022
        FM[Flow Matching 2022<br/>条件流匹配恒等式<br/>任意概率路径→速度回归]
        RF[Rectified Flow 2022<br/>reflow 让轨迹直线化<br/>少步采样误差↓]
    end

    subgraph 一步/少步 2023-2025
        CM[Consistency Models 2023<br/>学 任意t→x0 映射, 1 步生成]
        DMD[DMD 2023<br/>分布匹配蒸馏<br/>score 对齐 + 回归项]
        DMD2[DMD2 2024<br/>+真图对抗项, 质量↑稳定↑]
        MF[MeanFlow 2025<br/>时间平均速度场<br/>CM 与 FM 的统一, 1 步]
    end

    subgraph 架构演进
        UNET[U-Net 2021]
        DIT[DiT 2023<br/>Transformer 取代 U-Net<br/>并行友好（与 GPT 同构）]
        MMDIT[MMDiT 2024<br/>SD3/Flux 双流文本+图像]
        MOE[MoE 2025<br/>Wan2.2 双专家<br/>高/低噪声按 SNR 切换<br/>27B 总参/14B 激活]
    end

    GAN --> DDPM
    VAE --> DDPM
    DDPM --> SDE
    DDPM --> DDIM
    DDPM --> LDM
    SDE --> FM
    DDIM --> FM
    FM --> RF
    RF --> CM
    CM --> DMD
    DMD --> DMD2
    DMD2 --> MF
    RF --> MF
    LDM --> UNET
    UNET --> DIT
    DIT --> MMDIT
    MMDIT --> MOE
    FM --> MMDIT
    RF --> MOE
```

**读懂这张图的关键**：
- **Flow 路线**：FM 提出"预测速度场"，RF 让轨迹更直——
  这是 SD3/Flux/Wan 全部选择的主路线（少步采样）。
- **一步生成路线**：DMD 用分布匹配把多步 ODE 压缩成一步前向；
  MeanFlow（2025）从时间平均速度场出发统一 CM 与 FM。
- **架构线**：U-Net → DiT（并行友好）→ MMDiT（多模态）→ MoE（容量-成本解耦）。

---

## 2. 项目演进（本仓库 Phase 0-5 的落地）

```mermaid
flowchart TB
    P0[Phase 0 理论<br/>DDPM/FM/RF 手推公式<br/>4 篇笔记] --> P1
    P1[Phase 1 diffusers 跑通<br/>VAE→TE→UNet→scheduler<br/>tiny-sd + pokemon] --> P2
    P2[Phase 2 手写 mini flow<br/>33M UNet + DDPM/FM 双目标<br/>60k 步 × 2 模型] --> P3
    P3[Phase 3 DiT + 分布式<br/>TP/DP 等价性<br/>前向 0 误差, loss 差<1e-4] --> P4
    P4[Phase 4 MoE + DeepSpeed<br/>Wan2.2 读码 + 双专家实现<br/>ZeRO-2/3 跑通] --> P5
    P5[Phase 5 蒸馏一步生成<br/>1 步 FID 857→304<br/>DMD 分布匹配 +12%]
```

### 关键量化结果（每条都对应一个可复现实验）

| Phase | 实验 | 结果 | 结论 |
|---|---|---|---|
| 2 | flow vs ddpm（60k 步, CIFAR-10） | 50 步 FID **54.4** vs 94.1 | flow 少步优势实证（16 步 77.9 < ddpm 50 步 94.1）|
| 2 | EMA 陷阱诊断 | EMA FID 727 vs raw 94.6 | decay 0.9999 在快速训练下有害，评估必须 raw/EMA 双测 |
| 3 | TP=2 数学等价性 | 前向 **0 误差**，100 步 loss 差 **~1e-4** | 自写 DiT TP 切分正确 |
| 4 | MoE vs dense（15k 步） | FID 114 vs **181** | 同激活参数下强行分流有害；MoE 价值在容量-成本解耦 |
| 5 | 蒸馏（回归 + DMD） | 1 步 FID 857→**304** | 把 50 步 ODE 算力编码进一步网络 |

---

## 3. 演进逻辑总结（为什么这条路这样走）

```
质量/延迟权衡这条主线：
  多步 ODE（50 步, FID 54）      ← 训练上限, 质量优先
    ↓ 蒸馏（Phase 5）
  1 步生成（FID 304）            ← 部署, 延迟优先
    ↓ 下一步（未做, 可扩展）
  对抗/更强蒸馏（DMD2 式）       ← 1 步质量逼近多步

并行/显存主线：
  单卡（Phase 2）→ TP/DP（Phase 3, 等价性）→ ZeRO（Phase 4）
    ↓ 大模型方向（Wan2.2 路线）
  MoE 容量-成本解耦 + FSDP/Ulysses 序列并行
```
