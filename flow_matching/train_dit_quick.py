"""DiT + FM 短训练验证（单卡，5k 步出图）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from PIL import Image
from torchvision.utils import make_grid

from flow_matching.config import TrainConfig
from flow_matching.data import get_cifar10_loader
from flow_matching.dit import DiT
from flow_matching.objectives import FlowObjective
from flow_matching.sampler import sample_euler


def main():
    cfg = TrainConfig(max_steps=5000)
    torch.manual_seed(0)
    model = DiT(cfg.model).cuda()
    obj = FlowObjective(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500))
    dl = get_cifar10_loader(64, cfg.data_dir, num_workers=4)
    it = iter(dl)
    print(f"DiT params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)
    for step in range(5000):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.cuda()
        t = torch.rand(64, device="cuda")
        xt, z = obj.add_noise(x, t)
        v = x - z
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(model(xt, t), v)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        imgs = sample_euler(model, obj, (64, 3, 32, 32), steps=50,
                            device="cuda")
    os.makedirs("runs/phase3", exist_ok=True)
    grid = make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=8).clamp(0, 1)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(
        "uint8")).save("runs/phase3/dit-sample-5000.png")
    print("saved sample", flush=True)


if __name__ == "__main__":
    main()
