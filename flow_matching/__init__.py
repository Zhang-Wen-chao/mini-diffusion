"""mini-diffusion: 手写 diffusion/flow matching 训练（纯 PyTorch）

Phase 2 目标：
  1. DDPM（epsilon 目标）与 Flow Matching（velocity 目标）在同一个 UNet 上跑通
  2. 对比两种目标的训练动态与采样质量（步数 vs 质量）
  3. 全部零外部依赖（除 torch），风格对齐 mini-megatron
"""
