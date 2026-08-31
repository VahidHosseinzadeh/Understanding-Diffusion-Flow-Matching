"""Datasets. Two families, deliberately: 2D toys to see with, images to
scale to.

Everything is standardised to roughly zero mean and unit variance. That
is not cosmetic -- the noise endpoint of every path is N(0, I), so if
the data lives at a different scale the interpolation spends most of
its length somewhere neither endpoint resembles, and the velocity field
the model must learn gets needlessly large.
"""
from __future__ import annotations

import math
from pathlib import Path as _Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = _Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------
# 2D toy distributions -- start here
# --------------------------------------------------------------------

def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)


def two_moons(n: int = 8192, noise: float = 0.06) -> torch.Tensor:
    """Two interleaving crescents. Not linearly separable, and the two
    modes are close enough that a sloppy field bleeds mass between them."""
    n_out = n // 2
    n_in = n - n_out
    theta_out = torch.rand(n_out) * math.pi
    theta_in = torch.rand(n_in) * math.pi
    outer = torch.stack([theta_out.cos(), theta_out.sin()], dim=1)
    inner = torch.stack([1.0 - theta_in.cos(), 0.5 - theta_in.sin()], dim=1)
    x = torch.cat([outer, inner], dim=0) + noise * torch.randn(n, 2)
    return _standardize(x)


def eight_gaussians(n: int = 8192, std: float = 0.1, radius: float = 2.0) -> torch.Tensor:
    """Eight isolated modes on a circle. The cleanest test of mode
    coverage: dropped or smeared modes are obvious at a glance."""
    angles = torch.arange(8, dtype=torch.float32) * (2 * math.pi / 8)
    centers = torch.stack([angles.cos(), angles.sin()], dim=1) * radius
    idx = torch.randint(0, 8, (n,))
    x = centers[idx] + std * torch.randn(n, 2)
    return _standardize(x)


def spiral(n: int = 8192, noise: float = 0.05, turns: float = 1.5) -> torch.Tensor:
    """Two intertwined spiral arms -- a thin, curved, high-curvature
    manifold. Straight-line paths have the most trouble here, which
    makes it the interesting stress case for rectified flow."""
    t = torch.rand(n // 2).sqrt() * turns * 2 * math.pi
    r = t / (turns * 2 * math.pi)
    arm = torch.stack([r * t.cos(), r * t.sin()], dim=1)
    x = torch.cat([arm, -arm], dim=0)
    x = x + noise * torch.randn_like(x)
    return _standardize(x)


TOY_DATASETS = {"moons": two_moons, "eight_gaussians": eight_gaussians, "spiral": spiral}


def get_toy_dataloader(
    name: str = "moons",
    n: int = 8192,
    batch_size: int = 256,
    **kwargs,
) -> DataLoader:
    x = TOY_DATASETS[name](n=n, **kwargs)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------------------------------------
# Fashion-MNIST -- once the 2D case is clear
# --------------------------------------------------------------------

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),                        # -> [0, 1], (1, 28, 28)
        transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1]
    ]
)


def get_fashion_mnist(root: str | _Path = DATA_ROOT, train: bool = True):
    return datasets.FashionMNIST(root=str(root), train=train, download=True, transform=_TRANSFORM)


def get_image_dataloader(
    root: str | _Path = DATA_ROOT,
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
