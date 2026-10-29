#!/usr/bin/env python
"""Score a checkpoint: FID, KID, Inception Score, precision/recall.

    python dfm/evaluate.py --checkpoint runs/<run>/checkpoint.pt
    python dfm/evaluate.py --checkpoint ... --sampler euler heun --nfe 10 20 50

Like sample.py, the path, target and model come from the checkpoint;
what you choose is how to decode and how many samples to score. Every
(sampler, NFE) cell starts from the same seed, and budgets count network
calls rather than steps (heun spends two per step, one on its last), so
the cells of one table are a fair comparison.

Needs `pip install torch-fidelity`, and a GPU for anything past a smoke
test. The first run per split also pushes every real image through
Inception-v3; that is cached under data/fidelity_cache/ and reused.

FID is biased by sample count, so compare numbers only at equal --n.
The default 10k is for turnaround; papers report 50k (JiT's FID-50K).

A class-conditional checkpoint draws equally many samples of every
class, and --cfg-scale sweeps the guidance weight w (1 = plain
conditional, 0 = the same network's unconditional field):
    python dfm/evaluate.py --checkpoint ... --cfg-scale 0 1 1.5 2 3
--nfe still counts evaluations of the field; a guided one runs the
network twice, which each result records as `network_passes`.
"""
from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from checkpoint import load_checkpoint, sample_shape
from dataset import DATA_ROOT
from guidance import balanced_labels, network_passes
from metrics import METRICS, fashion_mnist_uint8, generate_images, image_metrics
from samplers import SAMPLERS, network_calls, steps_for_budget
from utils import get_device, seed_everything
from viz import save_image_grid

# Short names for the printed table; the JSON keeps torch-fidelity's.
_LABELS = {
    "frechet_inception_distance": "FID",
    "kernel_inception_distance_mean": "KID",
    "inception_score_mean": "IS",
    "precision": "precision",
    "recall": "recall",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sampler", nargs="+", choices=list(SAMPLERS), default=["euler"])
    p.add_argument("--nfe", nargs="+", type=int, default=[50],
                   help="network-call budgets; each sampler takes as many steps as fit")
    p.add_argument("--n", type=int, default=10_000, help="samples scored per cell")
    p.add_argument("--cfg-scale", nargs="+", type=float, default=[1.0],
                   help="classifier-free guidance weights w to sweep (class-conditional "
                        "checkpoints only): 1 = plain conditional, 0 = unconditional")
    p.add_argument("--guidance-interval", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("LO", "HI"), help="guide only while LO <= t <= HI (t=0 is noise)")
    p.add_argument("--reference", choices=["train", "test"], default="train",
                   help="the real split to compare against")
    p.add_argument("--metrics", nargs="+", choices=METRICS, default=["fid", "isc", "kid"])
    p.add_argument("--batch-size", type=int, default=500, help="generation batch size")
    p.add_argument("--no-ema", action="store_true", help="score raw weights instead of EMA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cache-dir", type=str, default=str(DATA_ROOT / "fidelity_cache"),
                   help="where the reference features and statistics are cached")
    p.add_argument("--out-dir", type=str, default=None,
                   help="where results go; default: beside the checkpoint")
    args = p.parse_args()

    # Fail before spending minutes generating samples, not after.
    try:
        fidelity_version = version("torch-fidelity")
    except PackageNotFoundError:
        raise SystemExit("torch-fidelity is not installed. Run: pip install torch-fidelity")

    seed_everything(args.seed)
    device = get_device(args.device)
    # Enough threads for the UNet and for Inception on CPU, without the
    # one-per-core oversubscription train.py warns about.
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    model, path, target, meta, epoch = load_checkpoint(
        args.checkpoint, device, use_ema=not args.no_ema)
    if meta["data"] != "fashion_mnist":
        raise SystemExit(f"these metrics score images; this checkpoint was trained on {meta['data']!r}")
    num_classes = meta.get("num_classes")
    if not num_classes and args.cfg_scale != [1.0]:
        raise SystemExit("--cfg-scale needs a class-conditional checkpoint (train.py --class-cond)")
    labels = balanced_labels(args.n, num_classes) if num_classes else None
    interval = tuple(args.guidance_interval)
    reference = fashion_mnist_uint8(train=args.reference == "train")
    reference_name = f"fashion_mnist-{args.reference}"
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Everything that changes the numbers, written into every result file,
    # so two files can be compared without guessing how each was produced.
    settings = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "weights": "raw" if args.no_ema else "ema",
        "n": args.n,
        "reference": reference_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "guidance_interval": list(interval) if num_classes else None,
        "torch_fidelity": fidelity_version,
        "meta": meta,
    }
    print(f"{path}  {target}  epoch={epoch}  weights={settings['weights']}  n={args.n}  "
          f"reference={reference_name}  device={device}")

    cells = [(name, budget, w) for name in args.sampler for budget in args.nfe
             for w in args.cfg_scale]
    for name, budget, w in cells:
        steps = steps_for_budget(name, budget)
        nfe = network_calls(name, steps)  # what was spent, not what was allowed
        passes = (network_passes(SAMPLERS[name], path, target, sample_shape(meta),
                                 steps, w, interval) if num_classes else nfe)
        # One file per cell, written as soon as the cell is done, and
        # named by every setting a rerun is likely to change: a later
        # run with other settings never overwrites it, and a crash an
        # hour into a sweep keeps everything finished so far.
        tag = ("_raw" if args.no_ema else "") + (f"_s{args.seed}" if args.seed else "")
        if num_classes:
            tag = f"_cfg{w:g}" + ("" if interval == (0.0, 1.0)
                                  else f"_int{interval[0]:g}-{interval[1]:g}") + tag
        stem = out_dir / f"eval_{args.reference}_n{args.n}_{name}_nfe{nfe}{tag}"

        samples = generate_images(model, path, target, SAMPLERS[name], args.n,
                                  sample_shape(meta), device, steps,
                                  batch_size=args.batch_size, seed=args.seed,
                                  labels=labels, guidance_scale=w,
                                  guidance_interval=interval)
        # What was actually scored: the first thing to look at when a
        # number seems off.
        # (With labels, one class per column.)
        shown = 10 * num_classes if num_classes else 64
        save_image_grid(samples[:shown].float() / 127.5 - 1.0, f"{stem}.png",
                        nrow=num_classes or 8)
        scores = image_metrics(samples, reference, reference_name, metrics=args.metrics,
                               cache_dir=args.cache_dir, verbose=True)

        Path(f"{stem}.json").write_text(json.dumps(
            {**settings, "sampler": name, "steps": steps, "nfe": nfe,
             "cfg_scale": w if num_classes else None, "network_passes": passes,
             "scores": scores},
            indent=2))
        cfg = f"w={w:<4g} passes={passes:<4} " if num_classes else ""
        print(f"{name:<15} nfe={nfe:<4} {cfg}" + "  ".join(
            f"{_LABELS[k]}={v:.4g}" for k, v in scores.items() if k in _LABELS)
            + f"   -> {stem}.json")


if __name__ == "__main__":
    main()
