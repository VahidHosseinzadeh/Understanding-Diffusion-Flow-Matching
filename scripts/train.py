#!/usr/bin/env python
"""Train a DDPM or flow-matching model on Fashion-MNIST.

Examples:
    python scripts/train.py --process ddpm --epochs 20
    python scripts/train.py --process flow_matching --epochs 20
    python scripts/train.py --process ddpm --schedule linear --max-steps 200 --subset 512  # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfm.data import get_dataloader
from dfm.ddpm import DDPM
from dfm.flow_matching import RectifiedFlow
from dfm.trainer import Trainer, TrainConfig
from dfm.unet import UNet
from dfm.utils import get_device, seed_everything


def build_process(args):
    if args.process == "ddpm":
        return DDPM(timesteps=args.timesteps, schedule=args.schedule)
    if args.process == "flow_matching":
        return RectifiedFlow()
    raise ValueError(f"unknown process: {args.process}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--process", choices=["ddpm", "flow_matching"], default="ddpm")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine", help="DDPM only")
    p.add_argument("--timesteps", type=int, default=1000, help="DDPM only")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", type=int, default=None, help="use only N training images (debugging)")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"device: {device}")

    dataloader = get_dataloader(batch_size=args.batch_size, subset=args.subset, num_workers=args.num_workers)
    model = UNet(base_channels=args.base_channels)
    process = build_process(args)

    out_dir = args.out_dir or f"runs/{args.process}"
    config = TrainConfig(epochs=args.epochs, lr=args.lr, out_dir=out_dir, max_steps=args.max_steps)
    trainer = Trainer(model, process, device, config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"process={args.process}  params={n_params/1e6:.2f}M  out_dir={out_dir}")
    trainer.fit(dataloader)
    print(f"done. checkpoint + sample grids in {out_dir}/")


if __name__ == "__main__":
    main()
