"""CIFAR-10 数据加载（HF datasets，走代理下载一次后缓存）"""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CIFAR10(Dataset):
    def __init__(self, split: str, data_dir: str):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),   # [-1, 1]
        ])
        self.ds = load_dataset("uoft-cs/cifar10", split=split, cache_dir=data_dir)
        self.ds.set_format("numpy")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.tf(self.ds[i]["img"].copy())   # 拷贝：HF numpy 数组不可写


def get_cifar10_loader(batch_size: int, data_dir: str, split: str = "train",
                       num_workers: int = 4):
    return DataLoader(
        CIFAR10(split, data_dir), batch_size=batch_size,
        shuffle=split == "train", num_workers=num_workers,
        drop_last=True, persistent_workers=num_workers > 0)
