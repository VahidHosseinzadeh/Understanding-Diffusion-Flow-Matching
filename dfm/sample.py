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
draws class k only, and several labels draw one column each:
    python dfm/sample.py --checkpoint ... --cfg-scale 3
    python dfm/sample.py --checkpoint ... --cfg-scale 3 --label 7
    python dfm/sample.py --checkpoint ... --cfg-scale 3 --label 0 5 7 --n 12

Li & He's own JiT checkpoints (checkpoint-last.pth) load as they are,
into their own model code (jit_pretrained.py). --tracker wandb also logs
the grid and the settings that drew it.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from checkpoint import load_checkpoint, sample_shape
from dataset import TOY_DATASETS
from guidance import ClassifierFreeGuidance, balanced_labels, network_passes
from samplers import SAMPLERS, network_calls
from tracking import make_tracker
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
    p.add_argument("--label", type=int, nargs="+", default=None,
                   help="draw only these classes, one per column (default: every class)")
    p.add_argument("--guidance-interval", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("LO", "HI"), help="guide only while LO <= t <= HI (t=0 is noise)")
    p.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA")
    p.add_argument("--trajectories", action="store_true", help="2D only: plot noise->data paths")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads; default 1 for the MLP (see train.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracker", choices=["none", "wandb"], default="none",
                   help="also log the grid and its settings; wandb is imported only if chosen")
    p.add_argument("--wandb-project", type=str, default="dfm")
    p.add_argument("--wandb-group", type=str, default="samples")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="default: the checkpoint's folder and the output file's name")
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

    calls = passes = network_calls(args.sampler, args.steps)
    nrow = 8
    if num_classes:
        if args.label is None:
            labels = balanced_labels(n, num_classes)
            nrow = min(num_classes, 10)  # one class per column, up to 10 of them
        else:
            labels = torch.tensor(args.label).repeat(-(-n // len(args.label)))[:n]
            nrow = len(args.label) if len(args.label) > 1 else 10
        interval = tuple(args.guidance_interval)
        model = ClassifierFreeGuidance(model, labels.to(device), args.cfg_scale, interval)
        passes = network_passes(sampler, path, target, sample_shape(meta), args.steps,
                                args.cfg_scale, interval)
        print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
              f"w={args.cfg_scale:g}  network calls={calls}  network passes={passes}")
    else:
        print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
              f"network calls={calls}")

    start = time.perf_counter()
    result = sampler(model, path, target, shape, device, steps=args.steps,
                     return_trajectory=args.trajectories and is_toy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(f"sampled {n} in {seconds:.1f}s")

    saved = {"samples": out}
    if args.trajectories and is_toy:
        save_trajectories(result, out)
        saved["samples"] = out.with_name(out.stem + "_samples.png")
        saved["trajectories"] = out
        save_scatter_2d(result[-1], saved["samples"])
    elif is_toy:
        save_scatter_2d(result, out)
    else:
        save_image_grid(result, out, nrow=nrow)
    print("saved " + " and ".join(str(f) for f in saved.values()))

    tracker = make_tracker(
        args.tracker,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or f"{Path(args.checkpoint).parent.name}_{out.stem}",
        config={"checkpoint": args.checkpoint, "sampler": args.sampler, "steps": args.steps,
                "n": n, "cfg_scale": args.cfg_scale, "guidance_interval": args.guidance_interval,
                "labels": args.label, "ema": not args.no_ema, "seed": args.seed, "meta": meta},
    )
    tracker.log_images(0, saved)
    tracker.log_scalars(0, {"network_calls": calls, "network_passes": passes, "seconds": seconds})
    tracker.finish()


if __name__ == "__main__":
    main()
