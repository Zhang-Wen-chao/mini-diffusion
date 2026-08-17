#!/usr/bin/env python3
"""Phase 1: Stable Diffusion 文本到图像微调（拆解版）

目的：完整跑通 AIGC 训练 pipeline（VAE → text encoder → UNet → scheduler），
理解每个组件的数据流。默认用 segmind/tiny-sd（小模型，L20 友好），
可用 --model-id 换成 stable-diffusion-v1-5。

用法（L20 容器，先 export 代理下载模型/数据）：
  python3 scripts/train_sd_finetune.py \
      --resolution 256 --train-batch-size 4 --max-train-steps 500 \
      --mixed-precision fp16 --save-every 250 --seed 0
"""
import argparse
import gc
import os

import torch
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_DEFAULT = "segmind/tiny-sd"          # 小模型：UNet 仅 99M，权重 ~1.3GB
DATASET_DEFAULT = "lambdalabs/pokemon-blip-captions"
VAE_SCALE = 0.18215                        # SD 家族固定的 latent 缩放系数


class TextImageDataset(Dataset):
    """prompt + 图像。注意：只在 rank0 下载/缓存数据集，accelerate 会广播。"""

    def __init__(self, dataset_name, resolution, tokenizer):
        self.ds = load_dataset(dataset_name, split="train")
        self.res = resolution
        self.tok = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # SD 图像归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        if isinstance(item["image"], Image.Image):
            image = item["image"].convert("RGB")
        else:
            image = Image.open(item["image"].filename).convert("RGB")
        if item.get("text"):
            caption = item["text"]
        else:
            caption = "a photo of a pokemon"
        return {"pixel_values": self.tf(image), "caption": caption,
                "input_ids": self.tok(
                    caption, max_length=77, padding="max_length", truncation=True,
                    return_tensors="pt")["input_ids"][0]}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_DEFAULT)
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--max-train-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mixed-precision", default="fp16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--output-dir", default="runs/sd-tiny")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=1, mixed_precision=args.mixed_precision)
    device = accelerator.device
    torch.manual_seed(args.seed)

    # ── 1. 模型组件（pipeline 四件套）────────────────────────────
    # VAE：像素空间(3×H×W) → latent 空间(4×H/8×W/8)，图像先压再扩散，省算力
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    # 文本编码：prompt → 77×768 embedding（CLIP），文本条件是 cross-attention 的 K/V
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    # UNet：扩散主体（含 cross-attention 注入文本）
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    # scheduler：噪声表 β_t + add_noise（前向）和去噪（采样）
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)              # 微调 UNet 即可，VAE/TE 冻结
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # VAE/TE 冻结在 fp32（unet 的 fp16 由 accelerate autocast 托管，
    # 避免 VAE 的 GroupNorm 在 fp16 下反向报 dtype 错误）

    # ── 2. 数据 ──────────────────────────────────────────────────
    train_ds = TextImageDataset(args.dataset, args.resolution, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size,
                          shuffle=True, num_workers=2, collate_fn=collate)

    # ── 3. 优化器（固定 8bit AdamW，省显存）──────────────────────
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=100, num_training_steps=args.max_train_steps)

    unet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dl, lr_scheduler)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # ── 4. 训练循环（与 LLM 训练的唯一区别就在目标函数）──────────
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dl:
            with accelerator.accumulate(unet):
                # VAE 编码到 latent 空间（冻结，fp32）
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * VAE_SCALE

                # 文本条件嵌入（冻结）
                text_emb = text_encoder(batch["input_ids"].to(device))[0]

                # 前向加噪：随机 t 时刻，x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, t)

                # 目标函数：预测噪声（DDPM 简化目标 L_simple）
                noise_pred = unet(noisy_latents, t,
                                  encoder_hidden_states=text_emb).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"step {global_step:5d} | loss {loss.item():.4f} | "
                      f"lr {lr_scheduler.get_last_lr()[0]:.2e} | "
                      f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

            if accelerator.is_main_process and global_step % args.save_every == 0 and global_step > 0:
                save_and_eval(args, unet, vae, text_encoder, tokenizer, global_step)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        save_and_eval(args, unet, vae, text_encoder, tokenizer, global_step)
    accelerator.end_training()


def save_and_eval(args, unet, vae, text_encoder, tokenizer, step):
    """保存 + 采样验证（DDIM 确定性采样）"""
    os.makedirs(args.output_dir, exist_ok=True)
    unet_saved = unet.module if hasattr(unet, "module") else unet
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet_saved, scheduler=DDIMScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None,
    ).to("cuda")
    pipe.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))

    prompts = ["a cute blue pokemon with red eyes",
               "a pokemon sitting in a field of flowers"]
    gen = torch.Generator("cuda").manual_seed(args.seed)
    for p in prompts:
        img = pipe(p, num_inference_steps=30, guidance_scale=7.5,
                   generator=gen).images[0]
        img.save(os.path.join(args.output_dir, f"sample-step{step}-{p[:20].replace(' ','_')}.png"))
    print(f"[saved] step {step} -> {args.output_dir}/checkpoint-{step}")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
