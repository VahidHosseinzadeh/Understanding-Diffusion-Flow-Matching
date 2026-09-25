#!/usr/bin/env python
"""Score a checkpoint: FID, KID, Inception Score, precision/recall.

    python dfm/evaluate.py --checkpoint runs/<run>/checkpoint.pt
    python dfm/evaluate.py --checkpoint ... --sampler euler heun --nfe 10 20 50

Like sample.py, the path, target and model come from the checkpoint;
what you choose is how to decode and how many samples to score. Every
(sampler, NFE) cell starts from the same seed, and budgets count network
calls rather than steps (heun spends two per step), so the cells of one
table are a fair comparison.

Needs `pip install torch-fidelity`, and a GPU for anything past a smoke
test. The first run per split also pushes every real image through
Inception-v3; that is cached under data/fidelity_cache/ and reused.

FID is biased by sample count, so compare numbers only at equal --n.
The default 10k is for turnaround; papers report 50k (JiT's FID-50K).
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
from metrics import METRICS, fashion_mnist_uint8, generate_images, image_metrics
from samplers import NFE_PER_STEP, SAMPLERS
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
                   help="network-call budgets; steps = nfe / calls per step")
    p.add_argument("--n", type=int, default=10_000, help="samples scored per cell")
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
        "torch_fidelity": fidelity_version,
        "meta": meta,
    }
    print(f"{path}  {target}  epoch={epoch}  weights={settings['weights']}  n={args.n}  "
          f"reference={reference_name}  device={device}")

    for name in args.sampler:
        for budget in args.nfe:
            steps = max(1, budget // NFE_PER_STEP[name])
            nfe = steps * NFE_PER_STEP[name]
            # One file per cell, written as soon as the cell is done, and
            # named by every setting a rerun is likely to change: a later
            # run with other settings never overwrites it, and a crash an
            # hour into a sweep keeps everything finished so far.
            tag = ("_raw" if args.no_ema else "") + (f"_s{args.seed}" if args.seed else "")
            stem = out_dir / f"eval_{args.reference}_n{args.n}_{name}_nfe{nfe}{tag}"

            samples = generate_images(model, path, target, SAMPLERS[name], args.n,
                                      sample_shape(meta), device, steps,
                                      batch_size=args.batch_size, seed=args.seed)
            # What was actually scored: the first thing to look at when a
            # number seems off.
            save_image_grid(samples[:64].float() / 127.5 - 1.0, f"{stem}.png")
            scores = image_metrics(samples, reference, reference_name, metrics=args.metrics,
                                   cache_dir=args.cache_dir, verbose=True)

            Path(f"{stem}.json").write_text(json.dumps(
                {**settings, "sampler": name, "steps": steps, "nfe": nfe, "scores": scores},
                indent=2))
            print(f"{name:<15} nfe={nfe:<4} " + "  ".join(
                f"{_LABELS[k]}={v:.4g}" for k, v in scores.items() if k in _LABELS)
                + f"   -> {stem}.json")


if __name__ == "__main__":
    main()
