# Phase 1 实验记录：SD 微调跑通（L20）

> 2026-08-15。目标：完整跑通 AIGC 训练 pipeline（VAE → text encoder → UNet → scheduler），
> 理解数据流 + 记录工程坑。

## 实验配置

| 项 | 值 |
|---|---|
| 模型 | segmind/tiny-sd（SD1.5 蒸馏版，UNet 99M，权重 ~1.3GB） |
| 数据集 | huggan/pokemon（公开，无 caption → 默认 prompt） |
| 分辨率 | 256px（VAE 后 latent 32×32） |
| batch / 精度 | 4 / fp16（accelerate mixed precision） |
| 步数 | 600，线性 LR 5e-5，梯度裁剪 1.0 |
| 显存 | 7.2 GB（L20 46GB 无压力） |
| loss | 0.13 → ~0.05-0.10（短微调，噪声较大） |
| 产物 | checkpoint-300/600 + 采样图（512×512 PNG，DDIM 30 步） |

## Pipeline 数据流（核心理解）

```
prompt ──CLIP tokenizer──> 77 tokens ──CLIPTextModel──> 77×768 embedding (冻结)
                                                              │
image ──Resize/CenterCrop/Normalize──> [3,256,256] ──VAE──> [4,32,32] latent ×0.18215
                                                              │
   noise = randn;  t ~ U[0,1000]
   noisy = add_noise(latent, noise, t) = √ᾱ_t·latent + √(1-ᾱ_t)·noise   ← 前向闭式
                                                              │
   noise_pred = UNet(noisy, t, encoder_hidden_states=77×768)   ← cross-attention 注入文本
   loss = MSE(noise_pred, noise)                                ← DDPM 简化目标
```

- **VAE 缩放 0.18215**：SD 家族固定的 latent 缩放（让 latent 方差接近 1，稳定训练）。
- **UNet 是唯一可训练组件**：微调只动 UNet，VAE/TE 冻结 → 显存大减。
- **采样**：DDIM（确定性 ODE 离散化，30 步），CFG=7.5（文本条件放大）。

## 踩坑记录（都是面试能讲的工程经验）

1. **transformers 5.9 与 diffusers 0.33 不兼容**：`FLAX_WEIGHTS_NAME` 被 transformers 5.x 移除。
   解法：升级 diffusers 到 0.39（兼容 5.x），**不动容器原有 transformers**。
   → 教训：NGC 镜像里的库版本很新，装生态库前先查兼容矩阵。
2. **`lambdalabs/pokemon-blip-captions` 是 gated 数据集**（要登录）。换 `huggan/pokemon`。
   → 教训：选公开数据集；脚本兜底无 caption 用默认 prompt。
3. **fp16 VAE 反向报 `Found dtype Half but expected Float`**：VAE 的 GroupNorm 在 fp16、
   autocast 范围外反向会炸。解法：**VAE/TE 保持 fp32，UNet fp16 由 accelerate 托管**。
   这也是官方 SD 训练脚本的默认做法（除非显存不够才用 fp16 VAE + 特判）。
4. **HF 下载走代理**：容器 `--net host`，代理在宿主机 mihomo `127.0.0.1:7892`
   （不是 handoff 里写的 7890，clash 进程没起、mihomo 在跑）。
   `export http_proxy=http://127.0.0.1:7892 https_proxy=...` 后下载正常。
5. **数据集 image 字段可能没有 `.filename`**（内存 PIL 对象），直接 `.convert("RGB")`。

## 结论

- Phase 1 验收 ✅：loss 下降 + 采样出图（DDIM 30 步）+ checkpoint 保存。
- 训练 600 步仅 7 分钟级（~50 步/分钟），L20 完全够用。
- 这套「VAE latent + 冻结 TE + 微调 UNet」就是 SD 微调/训练的标准骨架，
  Phase 2 手写 mini flow matching 时把 `add_noise/MSE(noise)` 换成
  `(1-t)x0+t·x1 / MSE(v, x1-x0)` 即可切到 flow 目标。

## 复现命令

```bash
cd /path/to/mini-diffusion-test
export http_proxy=http://127.0.0.1:7892 https_proxy=http://127.0.0.1:7892
python3 phase1/train_sd_tiny.py --dataset huggan/pokemon \
    --resolution 256 --train-batch-size 4 --max-train-steps 600 \
    --mixed-precision fp16 --save-every 300
```
