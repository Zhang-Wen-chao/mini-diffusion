# mini-diffusion

AIGC（Diffusion）分布式训练学习路线：从 mini-megatron（LLM 并行）到 AIGC 训练，
纯 PyTorch 从零实现 Diffusion 训练全链路。

覆盖知识体系：
**Diffusion 基础 → Flow Matching / Rectified Flow → 蒸馏（DMD/DMD2/MeanFlow）→ DiT/MoE → 分布式训练**。

## 快速开始

```bash
pip install -r requirements.txt

# 单元测试（CPU 即可，11+6 项）
python -m pytest flow_matching/tests/ -q

# 训练（CIFAR-10，自动下载缓存到 data/）
python -m flow_matching.train --objective flow --max-steps 30000 --mixed-precision bf16
python -m flow_matching.train --objective ddpm --max-steps 30000 --mixed-precision bf16

# 采样质量评估（FID，需先跑一次生成 ref-feats）
python -m flow_matching.eval_fid --objective flow --ckpt runs/ckpt-flow-final.pt \
    --steps-list 1 2 4 8 16 50 --ref-feats runs/real-feats.npy

# 分布式（TP=2 等价性验证）
NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=2 --master_port=299xx \
  -m flow_matching.train_dist --tp 2 --dp 1 --verify --max-steps 100
```

## 文档

- [docs/evolution.md](docs/evolution.md) — **技术演进路线图**（领域方法演进 + 项目 Phase 落地）
- [docs/reference.md](docs/reference.md) — 参考手册（论文清单 / LLM vs Diffusion 对比 / 技术栈）
- `docs/theory/` — Phase 0 理论笔记（DDPM ELBO 推导 → Flow Matching CFM 恒等式 → 蒸馏）
- `docs/phase{1..5}/` — 各 Phase 实验记录（含全部可复现数据与踩坑）

## 结构

```
mini-diffusion/
├── flow_matching/        # 核心教学包（纯 PyTorch，零外部依赖）
│   ├── model.py          # UNet（时间调制 GroupNorm + SiLU）
│   ├── dit.py            # DiT（patchify + adaLN-zero，TP 友好）
│   ├── moe_dit.py        # MoE DiT（双 FFN 专家 + timestep 路由，Wan2.2 范式）
│   ├── objectives.py     # DDPM（ε 预测）与 Flow Matching（v 预测）双目标
│   ├── sampler.py        # DDIM / Euler / ancestral 采样器
│   ├── parallel/         # TP/DP 并行原语（自写，不依赖 Megatron）
│   ├── train*.py         # 训练入口（单卡/分布式/蒸馏/DeepSpeed）
│   ├── eval*.py          # FID 评估与对比
│   └── tests/            # 单元测试 + TP 数学等价性验证
├── scripts/
│   └── train_sd_finetune.py   # Phase 1 学习脚本（diffusers 生态微调 SD）
└── docs/                 # 理论笔记 + 实验记录 + 演进图
```

## 关键结果（全部可复现）

| 实验 | 结果 |
|---|---|
| flow vs ddpm（60k 步, CIFAR-10） | 50 步 FID **54.4** vs 94.1（flow 少步优势） |
| DiT TP=2 数学等价性 | 前向 **0 误差**，100 步 loss 差 **~1e-4** |
| MoE vs dense（15k 步） | FID 114 vs 181（负结果：同激活参数下分流 FFN 有害） |
| 蒸馏一步生成 | 1 步 FID 857→**304**（DMD 分布匹配项 +12%） |
| DeepSpeed ZeRO-2/3 | 2 卡跑通，显存 1.6 GB/卡 |

## 约定

- 纯 PyTorch 实现，零外部依赖（除 torch），沿用 mini-megatron 风格
- 对比基线：diffusers / Megatron-Core / accelerate / DeepSpeed
- 数据与产物默认在 `data/` 与 `runs/`（已 gitignore）

## License

[MIT](LICENSE)
