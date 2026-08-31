#!/usr/bin/env python
"""Train a flow matching model.

Start in 2D, where you can see the whole velocity field:
    python dfm/train.py --data moons --epochs 40

Then images:
    python dfm/train.py --data fashion_mnist --model unet --epochs 20

Smoke test (seconds):
    python dfm/train.py --data moons --max-steps 50 --epochs 1
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import torch

from dataset import TOY_DATASETS, get_image_dataloader, get_toy_dataloader
from losses import T_SAMPLERS, interpolant_loss
from mlp import MLP
from paths import PATHS
from samplers import SAMPLERS
from targets import TARGETS
from tracking import TRACKERS, make_tracker
from trainer import Trainer, TrainConfig
from unet import UNet
from utils import get_device, seed_everything
from viz import save_image_grid, save_scatter_2d, save_velocity_field


def main():
    p = argparse.ArgumentParser()
    # the three axes
    p.add_argument("--path", choices=list(PATHS), default="linear")
    p.add_argument("--target", choices=list(TARGETS), default="velocity")
    p.add_argument("--sampler", choices=list(SAMPLERS), default="euler",
                   help="used only for periodic previews during training")
    # loss knobs
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--t-dist", choices=list(T_SAMPLERS), default="uniform")
    # data / model
    p.add_argument("--data", choices=list(TOY_DATASETS) + ["fashion_mnist"], default="moons")
    p.add_argument("--model", choices=["mlp", "unet"], default=None,
                   help="default: mlp for 2D data, unet for images")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=64)
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 2e-3 for the MLP, 2e-4 for the UNet")
    p.add_argument("--n-train", type=int, default=8192, help="toy datasets only")
    p.add_argument("--subset", type=int, default=None, help="image datasets only")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--preview-steps", type=int, default=50)
    p.add_argument("--preview-every-epochs", type=int, default=1)
    p.add_argument("--log-every-steps", type=int, default=10, help="tracker cadence")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads. Default: 1 for the MLP, min(8, cores) "
                        "for the UNet. On a many-core node the default of "
                        "one-thread-per-core makes small models ~1000x slower.")
    p.add_argument("--seed", type=int, default=0)
    # experiment tracking (optional; local PNGs are written regardless)
    p.add_argument("--tracker", choices=TRACKERS, default="none")
    p.add_argument("--wandb-project", type=str, default="dfm")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None,
                   help="default: <data>-<path>-<target>-s<seed>")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="tie a set of variations together, e.g. --wandb-group path-sweep")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    is_toy = args.data in TOY_DATASETS
    model_name = args.model or ("mlp" if is_toy else "unet")

    # A 2D MLP does dozens of microsecond-sized ops per step. Synchronising
    # one OpenMP thread per core across those costs far more than the work
    # itself -- on a 48-core node that is seconds per step instead of
    # milliseconds. Big convolutions do amortise the sync, so the UNet wants
    # several threads. Neither default serves both.
    threads = args.threads if args.threads is not None else (
        1 if model_name == "mlp" else min(8, os.cpu_count() or 1)
    )
    torch.set_num_threads(threads)

    # The MLP on 2D data tolerates a much larger step than a 6M-param UNet
    # on images; one default cannot serve both.
    lr = args.lr if args.lr is not None else (2e-3 if model_name == "mlp" else 2e-4)

    path = PATHS[args.path](sigma_min=args.sigma_min)
    target = TARGETS[args.target]()
    sampler = SAMPLERS[args.sampler]

    if is_toy:
        dataloader = get_toy_dataloader(args.data, n=args.n_train, batch_size=args.batch_size)
        reference = dataloader.dataset.tensors[0]
        model = MLP(dim=2, hidden=args.hidden, depth=args.depth)
        shape = (2048, 2)
    else:
        dataloader = get_image_dataloader(
            batch_size=args.batch_size, subset=args.subset, num_workers=args.num_workers
        )
        reference = None
        model = UNet(base_channels=args.base_channels)
        shape = (64, 1, 28, 28)

    loss_fn = partial(
        interpolant_loss, path=path, target=target, t_sampler=T_SAMPLERS[args.t_dist]
    )

    def preview(net, out_dir: Path, epoch: int) -> dict[str, Path]:
        """Write previews to disk and report them, so a tracker can mirror
        them. Panel names are stable across runs, which is what lets the
        same panel line up side by side when comparing variations."""
        samples = sampler(net, path, target, shape, device, steps=args.preview_steps, progress=False)
        produced: dict[str, Path] = {}

        samples_png = out_dir / f"samples_epoch{epoch:04d}.png"
        if is_toy:
            save_scatter_2d(samples, samples_png, reference=reference)
            field_png = out_dir / f"field_epoch{epoch:04d}.png"
            save_velocity_field(net, path, target, field_png, device, reference=reference)
            produced["velocity_field"] = field_png
        else:
            save_image_grid(samples, samples_png, nrow=8)
        produced["samples"] = samples_png
        return produced

    out_dir = args.out_dir or f"runs/{args.data}_{args.path}_{args.target}"
    config = TrainConfig(
        epochs=args.epochs, lr=lr, out_dir=out_dir, max_steps=args.max_steps,
        preview_every_epochs=args.preview_every_epochs,
        log_every_steps=args.log_every_steps,
    )
    meta = {
        "path": args.path, "sigma_min": args.sigma_min, "target": args.target,
        "model": model_name, "data": args.data, "hidden": args.hidden,
        "depth": args.depth, "base_channels": args.base_channels,
    }

    n_params = sum(q.numel() for q in model.parameters())
    print(f"device={device}  {path}  {target}  model={model_name} ({n_params/1e6:.2f}M)")
    print(f"data={args.data}  lr={lr:g}  threads={threads}  out_dir={out_dir}")

    # Every axis goes into the tracker's config, so runs can be grouped and
    # filtered by it later -- that is what makes "did the path help?"
    # answerable instead of guesswork.
    tracker = make_tracker(
        args.tracker,
        **(
            dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or f"{args.data}-{args.path}-{args.target}-s{args.seed}",
                group=args.wandb_group,
                tags=args.wandb_tags,
                out_dir=out_dir,
                config={**meta, **vars(config), "seed": args.seed,
                        "batch_size": args.batch_size, "t_dist": args.t_dist,
                        "n_params": n_params},
            )
            if args.tracker == "wandb"
            else {}
        ),
    )

    trainer = Trainer(model, loss_fn, device, config, preview_fn=preview,
                      meta=meta, tracker=tracker)
    try:
        trainer.fit(dataloader)
    finally:
        trainer.finish()
    print(f"done -- checkpoint, loss curve and previews in {out_dir}/")


if __name__ == "__main__":
    main()
