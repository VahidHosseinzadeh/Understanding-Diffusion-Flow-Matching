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

A class-conditional checkpoint (trained with --class-cond) draws a grid
with one class per column; --cfg-scale sets the guidance weight w
(1 = plain conditional, 0 = unconditional, >1 = guided), --label k
draws class k only:
    python dfm/sample.py --checkpoint ... --cfg-scale 3
    python dfm/sample.py --checkpoint ... --cfg-scale 3 --label 7
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from checkpoint import load_checkpoint, sample_shape
from dataset import TOY_DATASETS
from guidance import ClassifierFreeGuidance, balanced_labels, network_passes
from samplers import SAMPLERS, network_calls
from utils import get_device, seed_everything
from viz import save_image_grid, save_scatter_2d, save_trajectories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sampler", choices=list(SAMPLERS), default="euler")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n", type=int, default=None,
                   help="default: 2048 for 2D, 64 for images, 100 for class-conditional images")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="classifier-free guidance weight w: 1 = plain conditional, "
                        "0 = unconditional, >1 = guided. Class-conditional checkpoints only")
    p.add_argument("--label", type=int, default=None,
                   help="draw only this class (default: every class, one per column)")
    p.add_argument("--guidance-interval", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("LO", "HI"), help="guide only while LO <= t <= HI (t=0 is noise)")
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

    num_classes = meta.get("num_classes")
    if not num_classes and (args.label is not None or args.cfg_scale != 1.0):
        raise SystemExit("--label and --cfg-scale need a class-conditional checkpoint "
                         "(train.py --class-cond)")

    n = args.n or (2048 if meta["model"] == "mlp" else 100 if num_classes else 64)
    shape = (n, *sample_shape(meta))

    sampler = SAMPLERS[args.sampler]
    cfg_tag = f"_cfg{args.cfg_scale:g}" if num_classes else ""
    out = Path(args.out or f"{args.sampler}_{args.steps}steps{cfg_tag}.png")

    calls = network_calls(args.sampler, args.steps)
    nrow = 8
    if num_classes:
        labels = (torch.full((n,), args.label) if args.label is not None
                  else balanced_labels(n, num_classes))
        interval = tuple(args.guidance_interval)
        model = ClassifierFreeGuidance(model, labels.to(device), args.cfg_scale, interval)
        nrow = num_classes if args.label is None else 10
        passes = network_passes(sampler, path, target, sample_shape(meta), args.steps,
                                args.cfg_scale, interval)
        print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
              f"w={args.cfg_scale:g}  network calls={calls}  network passes={passes}")
    else:
        print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
              f"network calls={calls}")

    result = sampler(model, path, target, shape, device, steps=args.steps,
                     return_trajectory=args.trajectories and is_toy)

    if args.trajectories and is_toy:
        save_trajectories(result, out)
        save_scatter_2d(result[-1], out.with_name(out.stem + "_samples.png"))
        print(f"saved {out} and {out.with_name(out.stem + '_samples.png')}")
    else:
        if is_toy:
            save_scatter_2d(result, out)
        else:
            save_image_grid(result, out, nrow=nrow)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
