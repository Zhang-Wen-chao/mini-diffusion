# Phase 4 · Wan2.2 读码笔记：MoE 扩散 + FSDP/Ulysses

> 源码：github.com/Wan-Video/Wan2.2（Apache-2.0），2026-08-16 精读
> 配置：wan/configs/wan_t2v_A14B.py + wan/modules/model.py + wan/distributed/

## 1. MoE：双专家按 timestep 切换（本 phase 核心）

**不是 token 级路由，是"timestep 级路由"**——Wan2.2 把去噪过程按噪声水平切成两段，
每段一个**完整独立模型**：

```python
# wan_t2v_A14B.py
t2v_A14B.low_noise_checkpoint  = 'low_noise_model'   # 低噪声专家（细节）
t2v_A14B.high_noise_checkpoint = 'high_noise_model'  # 高噪声专家（布局）
t2v_A14B.boundary = 0.875                            # 切换阈值（timestep 比例）
```

```python
# text2video.py:186
if t.item() >= boundary:                 # t 大 = 噪声高 = 高噪声专家
    required_model_name = 'high_noise_model'
    offload_model_name  = 'low_noise_model'
# 不用的专家搬到 CPU 省显存（_prepare_model_for_timestep）
```

**设计动机**（README 版）：高噪声阶段负责整体布局、低噪声阶段负责细节——
两个"专精任务"分开训，总容量 27B 而每步只激活 14B。

**对训练的影响（关键 insight）**：双专家模型**分开训练**（各训各的数据范围），
不是联合训练！t∈[boundary,1] 的样本训 high_noise，t∈[0,boundary) 训 low_noise。
这避免了 batch 内 t 分流导致的负载不均衡和路由不稳定。
（我们教学版做法不同：单模型双 FFN + batch 内 mask 分流，见实验部分——两种都值得讲。）

## 2. Attention（Wan 的标志性细节）

```python
# model.py:101 WanSelfAttention
self.q = nn.Linear(dim, dim); self.k = ...; self.v = ...; self.o = ...
self.norm_q = WanRMSNorm(dim); self.norm_k = WanRMSNorm(dim)   # ← qk_norm
# forward: q/k 过 RMSNorm 再过 RoPE，flash_attention，window_size 控制窗口
```

- **qk_norm（RMSNorm 在 Q/K 上）**：Stable Diffusion 3 之后的主流做法，
  大幅提升训练稳定性（q/k 的数值范围被约束）。
- **RoPE + 3D 网格 freqs**：视频的 (F,H,W) 三维位置编码（`rope_apply(q, grid_sizes, freqs)`）。
- **window attention**：窗口注意力（`window_size`），A14B 用全局 (-1,-1)。
- 独立 q/k/v Linear（未融合）+ 独立 o。

## 3. DiT Block 调制（与我们的 mini 版一致）

```python
# model.py:217
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
x = x + self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0]) * e[2]   # adaLN
```

6 路调制（shift/scale/gate × 2），与我们的 adaLN-zero 同源
（Wan 用可学习 modulation 偏置而非零初始化——小差异）。

## 4. 分布式（推理多卡：FSDP + Ulysses）

```python
# distributed/fsdp.py
FSDP(module=model, sharding_strategy=FULL_SHARD,
     auto_wrap_policy=lambda_fn=lambda m: m in model.blocks,   # 按 block 包装
     mixed_precision=BF16, sync_module_states=True)
```

- **FSDP FULL_SHARD**：参数/梯度/优化器状态全部切分，**权重在 CPU 内存也省**。
- 按 `model.blocks` 包装：每个 block 独立 all-gather，通信-计算重叠更细粒度。
- `sync_module_states=True`：rank0 加载权重后广播（避免每 rank 都读盘）。

```python
# distributed/ulysses.py
q = all_to_all(q, scatter_dim=2, gather_dim=1)   # 序列维 ↔ 头维
x = all_to_all(x, scatter_dim=1, gather_dim=2)   # 注意力后再换回来
x = flash_attention(q, k, v, ...)
```

- **Ulysses（arXiv:2309.14509）序列并行**：注意力前把"序列维切分"转成"头维切分"
  （all_to_all），每 rank 处理全部序列的部分 head；注意力后再转回。
- 好处：注意力阶段**零通信**（head 维独立），只有两次 all_to_all。
- 与 FSDP 组合 = Wan 官方 8 卡推理方案。

## 5. 训练目标（与我们的 mini 一致）

- **Flow Matching / rectified flow**：Wan2.1 技术报告（arXiv:2503.20314），
  sigma shift = 12（`sample_shift`，时间采样偏向中间噪声段）。
- 40 步采样（sample_steps），boundary=0.875 → t 的前 12.5% 用高噪声专家。
- VAE 2.2：T×H×W 压缩 4×16×16（TI2V-5B 再 patchify 到 4×32×32）。

## 6. 对我们的启发（Phase 4 实验设计）

| Wan2.2 | 我们的教学版 |
|---|---|
| 双**完整模型**按 timestep 切换 | 单模型内双 **FFN 专家**（共享 attention） |
| 分开训练 | 联合训练 + batch 内 mask 分流 |
| boundary=0.875（40 步采样） | boundary=0.5（FM t∈[0,1]） |
| FSDP + Ulysses（推理） | 复用 Phase 3 的 TP/DP |
| qk_norm + RoPE + window attn | adaLN-zero + sdpa（迷你版） |

> 核心可迁移点：**"按 timestep 分工"的 MoE 范式**——高噪声学布局、低噪声学细节。
> 我们验证：同样激活参数量下，双专家是否比单专家（dense）质量更好。
