# 04 · 并行复用点：mini-megatron → diffusion 训练的施工图

> 目的：明确 Phase 3（分布式 + DiT）哪些代码直接复用、哪些要改、哪些是新写。
> 结论先行：**训练循环骨架与并行原语 100% 复用，diffusion 新增的是目标函数层与采样层。**

## 1. 复用对照表

| mini-megatron 组件 | 在 diffusion 里的角色 | 复用方式 |
|---|---|---|
| `parallel/process_groups.py` TP/PP/DP 组构建 | 完全一样 | 直接拷贝 |
| `parallel/tensor_parallel.py` Column/Row Linear | DiT/UNet 的 Linear 层 | 直接复用（换成 DiT 层结构） |
| `parallel/pipeline_parallel.py` 1F1B + 梯度累积 | 模型层切分 | 直接复用（层数=DiT blocks） |
| `parallel/data_parallel.py` 梯度 all-reduce | 完全一样 | 直接拷贝 |
| `trainer.py` 训练循环 + AMP + clip | 循环骨架 | 修改：目标函数换 FM loss，无 labels |
| `comm/` all-reduce/send-recv/overlap | 完全一样 | 直接拷贝 |
| `model/transformer.py` attention/MLP | DiT 复用其中 attention/FFN | 修改：加时间嵌入 + adaLN 调制 |
| `model/embedding.py` | 不需要（无词表） | 删除，换成 patchify + VAE 接口 |

## 2. 训练循环对比（伪代码级）

```
LLM (mini-megatron 已有):              Diffusion (Phase 2 手写, Phase 3 并行化):
  x, labels = batch                     x1 = batch (图像/视频 latent)
  h = model(x)                          t = sample_timestep()          # 新增
  loss = CE(h, labels)                  x0 ~ N(0, I)                    # 新增
  loss.backward()                       xt = (1-t)x0 + t*x1             # 新增
  grad_sync (DP) / step                  v = model(xt, t, cond)         # 加时间条件
                                        loss = MSE(v, x1 - x0)          # 换目标
                                        loss.backward() / grad_sync / step
```

并行化差异点：
- 前向/反向/梯度同步/优化器步：**完全相同的代码路径**，TP/PP/DP 分组逻辑不动。
- PP 下 loss 在最后 stage 算（和 LLM 一样），但 FM loss 不需要 labels 广播（更简单）。
- **新增压力点：长序列**。图像/视频 latent 序列 = H/8 × W/8 × F/4（16×16×4 VAE），
  720p 视频可达 10K+ token。序列并行（SP）/ activation checkpointing 是 Phase 3 的优化项。

## 3. DiT 的并行友好性

- DiT 是无 attention 位置编码的纯 transformer block（adaLN-zero 调制），
  每个 block 就是 attention + FFN → **TP 切分与 GPT 完全同构**（QKV/FFN 列切 + 行切）。
- 时间嵌入 $\mathbf t$ 和文本条件 $\mathbf c$ 只做 block 级调制（缩放/偏置），
  跨 rank 复制即可（无需通信）。
- 结论：DiT 是"并行最好做"的扩散架构，这也是它取代 UNet 的原因之一。

## 4. 复用清单（Phase 3 开工时逐项勾）

```
comm/           all_reduce.py send_recv.py overlap_tp.py      ✅ 拷贝
parallel/       process_groups.py tensor_parallel.py
                pipeline_parallel.py data_parallel.py          ✅ 拷贝
trainer.py      optimizer/AMP/clip/logger                      ✅ 改目标函数
model/          attention.py mlp.py (抽取) → DiT blocks       ⚠️ 重构
新增:            patchify.py vae_loader.py timestep_embed.py
                flow_loss.py sampler.py (ODE/euler)
```

## 5. 一句话总结

> Phase 3 的施工量 = 复用 mini-megatron 的并行骨架（不动）+ 手写 diffusion 层
> （目标函数/时间条件/采样器/DiT block）+ 新增长序列优化（SP/重计算）。
