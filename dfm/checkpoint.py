"""Rebuild a trained model from its checkpoint.

Path, target and architecture are read back from the `meta` block that
`Trainer` writes, so decoding never depends on retyping the flags a run
was trained with -- a mismatch there produces silently wrong samples,
not an error. sample.py and evaluate.py both load through here, so
there is exactly one place that can get it wrong.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn

from mlp import MLP
from paths import PATHS, Path
from targets import TARGETS, Target
from unet import UNet


class Loaded(NamedTuple):
    model: nn.Module          # in eval mode, on the requested device
    path: Path
    target: Target
    meta: dict[str, Any]      # the run's configuration, as train.py wrote it
    epoch: int | None         # which point in training these weights are from


def load_checkpoint(checkpoint: str, device: torch.device, use_ema: bool = True) -> Loaded:
    """Load weights plus the path/target they were trained with.

    EMA weights by default: they are what the training previews show,
    and what you would ship.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    if not meta:
        raise SystemExit(
            "checkpoint has no `meta` block -- it predates config-in-checkpoint "
            "and cannot be reconstructed unambiguously. Retrain it."
        )

    if meta["model"] == "mlp":
        model = MLP(dim=2, hidden=meta["hidden"], depth=meta["depth"])
    else:
        model = UNet(base_channels=meta["base_channels"])
    model.load_state_dict(ckpt["ema" if use_ema else "model"])

    return Loaded(
        model=model.to(device).eval(),
        path=PATHS[meta["path"]](beta_min=meta.get("beta_min", 0.0)),
        target=TARGETS[meta["target"]](),
        meta=meta,
        epoch=ckpt.get("epoch"),
    )


def sample_shape(meta: dict[str, Any]) -> tuple[int, ...]:
    """The shape of one sample, without the batch dimension."""
    return (2,) if meta["model"] == "mlp" else (1, 28, 28)
