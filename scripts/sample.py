#!/usr/bin/env python
"""Sample from a checkpoint.

The path/target/model are read back from the checkpoint's `meta`, so
the only thing you choose here is how to *decode* -- which is the point:
one trained model, many samplers.

    python scripts/sample.py --checkpoint runs/moons_linear_velocity/checkpoint.pt
    python scripts/sample.py --checkpoint ... --sampler heun --steps 5

Compare solvers at equal network calls, not equal steps (Heun uses two
per step):
    python scripts/sample.py --checkpoint ... --sampler euler --steps 10
    python scripts/sample.py --checkpoint ... --sampler heun  --steps 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dfm.data import TOY_DATASETS
from dfm.mlp import MLP
from dfm.paths import PATHS
from dfm.samplers import SAMPLERS
from dfm.targets import TARGETS
from dfm.unet import UNet
from dfm.utils import get_device, seed_everything
from dfm.viz import save_image_grid, save_scatter_2d, save_trajectories


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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    if not meta:
        raise SystemExit(
            "checkpoint has no `meta` block -- it predates config-in-checkpoint "
            "and cannot be reconstructed unambiguously. Retrain it."
        )

    path = PATHS[meta["path"]](sigma_min=meta.get("sigma_min", 0.0))
    target = TARGETS[meta["target"]]()
    is_toy = meta["data"] in TOY_DATASETS

    if meta["model"] == "mlp":
        model = MLP(dim=2, hidden=meta["hidden"], depth=meta["depth"])
        n = args.n or 2048
        shape = (n, 2)
    else:
        model = UNet(base_channels=meta["base_channels"])
        n = args.n or 64
        shape = (n, 1, 28, 28)

    model.load_state_dict(ckpt["model" if args.no_ema else "ema"])
    model.to(device).eval()

    sampler = SAMPLERS[args.sampler]
    out = Path(args.out or f"{args.sampler}_{args.steps}steps.png")

    print(f"{path}  {target}  sampler={args.sampler}  steps={args.steps}  "
          f"network calls={args.steps * (2 if args.sampler == 'heun' else 1)}")

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
