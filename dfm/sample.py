#!/usr/bin/env python
"""Sample from a checkpoint.

The path/target/model are read back from the checkpoint's `meta`, so
the only thing you choose here is how to *decode* -- which is the point:
one trained model, many samplers.

    python dfm/sample.py --checkpoint runs/moons_linear_velocity/checkpoint.pt
    python dfm/sample.py --checkpoint ... --sampler heun --steps 5

Compare solvers at equal network calls, not equal steps (Heun uses two
per step, one on its last):
    python dfm/sample.py --checkpoint ... --sampler euler --steps 9
    python dfm/sample.py --checkpoint ... --sampler heun  --steps 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from checkpoint import load_checkpoint, sample_shape
from dataset import TOY_DATASETS
from samplers import SAMPLERS, network_calls
from utils import get_device, seed_everything
from viz import save_image_grid, save_scatter_2d, save_trajectories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sampler", choices=list(SAMPLERS), default="euler")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n", type=int, default=None, help="default: 2048 for 2D, 64 for images")
    p.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA")
    p.add_argument("--trajectories", action="store_true", help="2D only: plot noise->data paths")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads; default 1 for the MLP (see train.py)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    model, path, target, meta, _ = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    is_toy = meta["data"] in TOY_DATASETS

    torch.set_num_threads(
        args.threads if args.threads is not None
        else (1 if meta["model"] == "mlp" else min(8, os.cpu_count() or 1))
    )

    n = args.n or (2048 if meta["model"] == "mlp" else 64)
    shape = (n, *sample_shape(meta))

    sampler = SAMPLERS[args.sampler]
    out = Path(args.out or f"{args.sampler}_{args.steps}steps.png")

    print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
          f"network calls={network_calls(args.sampler, args.steps)}")

    result = sampler(model, path, target, shape, device, steps=args.steps,
                     return_trajectory=args.trajectories and is_toy)

    if args.trajectories and is_toy:
        save_trajectories(result, out)
        save_scatter_2d(result[-1], out.with_name(out.stem + "_samples.png"))
        print(f"saved {out} and {out.with_name(out.stem + '_samples.png')}")
    else:
        (save_scatter_2d if is_toy else save_image_grid)(result, out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
