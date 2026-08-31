"""Tracking must be strictly optional: the trainer behaves identically
with no tracker, and never depends on `wandb` being importable.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch

from dfm.losses import interpolant_loss
from dfm.mlp import MLP
from dfm.paths import LinearPath
from dfm.targets import VelocityTarget
from dfm.tracking import NullTracker, Tracker, make_tracker
from dfm.trainer import Trainer, TrainConfig


class RecordingTracker(Tracker):
    """Stands in for wandb so the wiring is testable without a network."""

    def __init__(self):
        self.scalars: list[tuple[int, dict]] = []
        self.images: list[tuple[int, dict]] = []
        self.finished = False

    def log_scalars(self, step, values):
        self.scalars.append((step, values))

    def log_images(self, step, images):
        self.images.append((step, images))

    def finish(self):
        self.finished = True


def _tiny_setup(tmp_path: Path, tracker=None, **cfg):
    model = MLP(dim=2, hidden=16, depth=2)
    loss_fn = partial(interpolant_loss, path=LinearPath(), target=VelocityTarget())
    config = TrainConfig(epochs=1, out_dir=str(tmp_path), **cfg)
    data = torch.utils.data.TensorDataset(torch.randn(64, 2))
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    return model, loss_fn, config, loader


def test_null_tracker_is_the_default_and_does_nothing(tmp_path):
    model, loss_fn, config, loader = _tiny_setup(tmp_path)
    trainer = Trainer(model, loss_fn, torch.device("cpu"), config)
    assert isinstance(trainer.tracker, NullTracker)
    trainer.fit(loader)  # must not raise
    assert (tmp_path / "checkpoint.pt").exists()


def test_scalars_and_preview_images_reach_the_tracker(tmp_path):
    rec = RecordingTracker()
    model, loss_fn, config, loader = _tiny_setup(tmp_path, log_every_steps=2)

    def preview(net, out_dir: Path, epoch: int) -> dict[str, Path]:
        p = out_dir / f"preview_{epoch}.png"
        p.write_bytes(b"not-a-real-png")
        return {"samples": p}

    trainer = Trainer(model, loss_fn, torch.device("cpu"), config,
                      preview_fn=preview, tracker=rec)
    trainer.fit(loader)
    trainer.finish()

    assert rec.scalars, "expected throttled loss logging"
    assert all("loss" in v for _, v in rec.scalars)
    assert all(step % 2 == 0 for step, _ in rec.scalars)
    assert rec.images and "samples" in rec.images[0][1]
    assert rec.finished


def test_preview_returning_none_is_allowed(tmp_path):
    """Local-only previews need not report anything."""
    rec = RecordingTracker()
    model, loss_fn, config, loader = _tiny_setup(tmp_path)
    trainer = Trainer(model, loss_fn, torch.device("cpu"), config,
                      preview_fn=lambda net, d, e: None, tracker=rec)
    trainer.fit(loader)
    assert rec.images == []


def test_make_tracker_rejects_unknown_backend():
    assert isinstance(make_tracker("none"), NullTracker)
    with pytest.raises(ValueError, match="unknown tracking backend"):
        make_tracker("tensorboard")
