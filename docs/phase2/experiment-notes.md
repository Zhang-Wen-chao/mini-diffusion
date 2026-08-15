# Phase 2 实验记录：mini flow matching（DDPM vs Flow 对比）

> 2026-08-15。目标：手写 DDPM 与 Flow Matching 双目标，同一 UNet 上对比训练动态与采样质量。
> 代码：`flow_matching/`（纯 PyTorch，零 diffusers 依赖），33M 参数 UNet，CIFAR-10。

## 实验配置

| 项 | 值 |
|---|---|
| 模型 | 自写 UNet 33M（GroupNorm+SiLU，时间调制 pre-modulation，8×8/4×4 加 attention） |
| 数据 | CIFAR-10 32×32（HF 下载，缓存到 data/） |
| 训练 | 60k 步 × BS64，AdamW lr=2e-4 (5k warmup)，BF16 autocast，grad clip 1.0 |
| 目标 | ddpm: ε 预测 (线性 β 1e-4→2e-2)；flow: v 预测 (x_t=(1-t)z+t·x, t~U[0,1]) |
| 采样 | ddpm: DDIM (确定性)；flow: Euler（midpoint 规则） |
| 硬件 | 1×L20，~1200 img/s（~19 步/秒），60k 步 ≈ 55 分钟 |
| 评估 | InceptionV3 特征 FID（2048 样本） |

## 结果 1：训练动态（loss 曲线）

| 目标 | 30k 步 loss | 60k 步 loss |
|---|---|---|
| ddpm (ε) | 0.023 | ~0.020 |
| flow (v) | 0.18 | ~0.18 |

> 注：两个 loss 数值**不可直接比**（预测对象不同：ε 单位方差 vs v 方差 ~4）。
> 各自收敛到稳态即正常。

## 结果 2：采样质量（FID，越小越好）

### flow（raw 权重，Euler 采样）

| 步数 | 1 | 2 | 4 | 8 | 16 | 50 |
|---|---|---|---|---|---|---|
| FID (30k) | 701 | 592 | 290 | 236 | 204 | 178 |
| FID (60k) | 860 | 508 | 250 | 122 | **77.9** | **54.4** |

### ddpm（raw 权重，DDIM 采样）

| 步数 | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| FID (30k, EMA) | 302 | 265 | 506 | 727 | 849 |
| FID (30k, raw) | - | 248 | - | **94.6** | 109 |
| FID (60k, raw) | 661 | 321 | 164 | **94.1** | 120 |

## 核心结论（60k 最终版）

1. **同训练量下 flow 完胜 ddpm**：50 步 FID 54.4 vs 94.1。
2. **flow 少步优势实证**：flow 16 步 (77.9) < ddpm 50 步 (94.1)——**用一半不到的步数达到更好质量**，
   这正是 SD3/Flux/Wan 全部改用 flow 路线的工程动机。
3. **flow 1 步仍差（860）**：直线插值只是"更直"，真正 1 步生成要靠蒸馏/reflow
   （Phase 5 DMD/MeanFlow 的动机）。
4. **flow 在 30k→60k 大幅改善（178→54.4），ddpm 停滞（94.6→94.1）**：
   flow 目标收敛更慢但持续改善；ddpm 可能饱和。长训练下差距进一步拉大。
5. **ddpm 100 步 (120) > 50 步 (94)**：大 t 区欠拟合持续影响，步数多了反而经过更多
   低噪声区（模型在该区 ε 预测非零，见发现 2）。

## 关键发现（都是面试素材）

### 发现 1：EMA 陷阱（本次实验最大的坑）
- 症状：训练 loss 正常收敛，但采样质量灾难性差（ddpm 50 步 FID 727）。
- 定位：EMA 权重 vs raw 权重对比采样 → **EMA 全灰（std 0.145），raw 正常（std 0.44）**。
- 根因：decay=0.9999 → EMA 半衰期 ~6931 步。模型在训练末期仍在快速改善
  （lr 恒定 2e-4，无 decay），EMA 严重滞后于当前权重。
- 教训：**小规模快速实验里 EMA 不是免费的**。评估必须同时测 raw 和 EMA，
  或用更小的 decay（0.999）+ 训练后期再做 EMA。
- 诊断方法论：① raw vs EMA 各采样一次比 std/饱和率；② 逐层扫权重差异；
  ③ 跟踪采样路径中间状态找分叉点。**先用 2 分钟诊断，不要凭感觉猜。**

### 发现 2：大 t 区欠拟合 + t=0 输出非零
- t=999 纯噪声输入下，模型 ε 预测能量 ~1.0（理想 ≈3.0，即"输出≈输入"）；
  t=0 干净输入下 ε 预测能量 ~2.3（理想 ≈0）。
- 结论：30k 步对 CIFAR 不够，模型在极端 t 区（0 和 999）没有学好；
  raw 50 步 FID 94.6 已可接受，成熟训练应 <30。续训到 60k 步看改善幅度。

### 发现 3：flow 少步优势（60k 实证）
- flow 16 步 FID 77.9 < ddpm 50 步 FID 94.1：**1/3 步数，更好质量**。
- flow 1 步 860：直线路径不够直 → 一步生成需要蒸馏（Phase 5）。
- flow 30k→60k 改善 3.3x，ddpm 停滞：flow 收敛慢但上限更高。

## 踩坑记录

1. **UNet 通道数设计 bug**：UpBlock 首个 ResBlock 输入应为 below+skip 通道和，
   不是 2×ch（网格 std 直接崩）。测试先行。
2. **ResBlock 调制只作用于 norm1（cin 维）**：norm2 是 cout 维，用同一组 scale/shift 会维度错。
3. **pkill -f 自杀**：命令串里含匹配模式会杀掉自己所在的 bash（exit 143），
   用 `[.]` 转义或精确 PID。
4. **nohup 输出缓冲**：python 重定向到文件要加 `-u`，否则看不到 step 日志。
5. **数据集/模型下载**：gated 数据集（lambdalabs/pokemon-blip-captions）要登录，
   换公开数据集；HF 下载需代理（宿主机 mihomo 7892）。
6. **resume 支持**：`--resume` 参数必须进 TrainConfig（曾漏传）。

## 复现命令

```bash
# 训练（resume 到 60k）
python3 -u -m flow_matching.train --objective flow --max-steps 60000 \
    --mixed-precision bf16 --out-dir runs/phase2/flow \
    --resume runs/phase2/flow/ckpt-flow-final.pt
python3 -u -m flow_matching.train --objective ddpm --max-steps 60000 \
    --mixed-precision bf16 --out-dir runs/phase2/ddpm \
    --resume runs/phase2/ddpm/ckpt-ddpm-final.pt

# FID 评估（raw 权重）
CUDA_VISIBLE_DEVICES=1 python3 -u -m flow_matching.eval_fid_weights \
    --objective flow --ckpt runs/phase2/flow/ckpt-flow-final.pt --weights model \
    --steps-list 1 2 4 8 16 50 --ref-feats runs/phase2/real-feats.npy
CUDA_VISIBLE_DEVICES=1 python3 -u -m flow_matching.eval_fid_weights \
    --objective ddpm --ckpt runs/phase2/ddpm/ckpt-ddpm-final.pt --weights model \
    --steps-list 5 10 25 50 100 --ref-feats runs/phase2/real-feats.npy

# 单测
python3 -m pytest flow_matching/tests/ -q   # 11 passed
```
