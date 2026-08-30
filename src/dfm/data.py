"""Fashion-MNIST data loading.

Images are scaled to [-1, 1], which is the convention both the DDPM
and flow-matching processes in this repo assume (Gaussian noise and
data then live on comparable scales).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),  # -> [0, 1], shape (1, 28, 28)
        transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1]
    ]
)


def get_fashion_mnist(root: str | Path = DATA_ROOT, train: bool = True):
    return datasets.FashionMNIST(root=str(root), train=train, download=True, transform=_TRANSFORM)


def get_dataloader(
    root: str | Path = DATA_ROOT,
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 2,
    subset: int | None = None,
) -> DataLoader:
    ds = get_fashion_mnist(root=root, train=train)
    if subset is not None:
        ds = torch.utils.data.Subset(ds, range(subset))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
    )


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
