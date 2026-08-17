# mini-diffusion

AIGC 分布式训练学习路线：从 mini-megatron（LLM 并行）到 Diffusion 训练。

覆盖知识体系：
**Diffusion 基础 → Flow Matching / Rectified Flow → 蒸馏（DMD/DMD2/MeanFlow）→ DiT/MoE → 分布式训练**。

## 文档

- [docs/reference.md](docs/reference.md) — 参考手册（论文清单 / LLM vs Diffusion 对比 / 技术栈）
- [docs/evolution.md](docs/evolution.md) — **技术演进路线图**（领域方法演进 + 项目 Phase 落地）
- `docs/` — 论文笔记与理论推导

## Phase 状态

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 论文精读（DDPM / Flow Matching / Rectified Flow） | ✅ docs/theory/01-04 |
| 1 | diffusers 跑通 SD 训练 | ✅ docs/phase1/experiment-notes.md |
| 2 | 手写 mini flow matching（纯 PyTorch） | ✅ flow_matching/ + docs/phase2/ |
| 3 | DiT + 分布式（TP/DP，等价性验证通过） | ✅ docs/phase3/experiment-notes.md |
| 4 | MoE + DeepSpeed（Wan2.2 路线，负结果分析） | ✅ docs/phase4/ |
| 5 | 蒸馏一步生成（1 步 FID 857→304，DMD 增强 +12%） | ✅ docs/phase5/ |
| 4 | MoE + DeepSpeed（Wan2.2 架构） | ⏳ |
| 5 | 蒸馏（DMD / MeanFlow） | ⏳ |

## 约定

- 纯 PyTorch 实现，零外部依赖（除 torch），沿用 mini-megatron 风格
- 对比基线：diffusers / Megatron-Core / accelerate
- 实验环境：L20 4 卡（`NGC PyTorch 容器` 容器），沿用 4×L20 基准方法论
