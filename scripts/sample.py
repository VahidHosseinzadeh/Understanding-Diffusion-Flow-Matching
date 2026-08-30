#!/usr/bin/env python
"""Generate a grid of samples from a trained checkpoint.

Example:
    python scripts/sample.py --checkpoint runs/ddpm/checkpoint.pt --process ddpm --out grid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dfm.ddpm import DDPM
from dfm.flow_matching import RectifiedFlow
from dfm.unet import UNet
from dfm.utils import get_device, save_image_grid, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--process", choices=["ddpm", "flow_matching"], default="ddpm")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampler", choices=["ancestral", "ddim"], default="ancestral", help="DDPM only")
    p.add_argument("--steps", type=int, default=50, help="DDIM / flow-matching step count")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--out", type=str, default="samples.png")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    model = UNet(base_channels=args.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["ema"] if args.use_ema else ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    if args.process == "ddpm":
        process = DDPM(timesteps=args.timesteps, schedule=args.schedule)
        shape = (args.n, 1, 28, 28)
        if args.sampler == "ddim":
            samples = process.ddim_sample(model, shape, device, steps=args.steps)
        else:
            samples = process.sample(model, shape, device)
    else:
        process = RectifiedFlow()
        shape = (args.n, 1, 28, 28)
        samples = process.sample(model, shape, device, steps=args.steps)

    save_image_grid(samples, args.out, nrow=8)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
