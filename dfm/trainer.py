"""A training loop that knows nothing about flow matching.

It takes a `loss_fn(model, batch) -> scalar` and calls it. Which path,
target, t-distribution and weighting produced that closure is the
caller's business. Adding a method later therefore never touches this
file -- which is the test of whether the axis split is real.

Previews are injected the same way, via `preview_fn`, so the trainer is
not tied to images. That matters: the old version hardcoded a
(B, 1, 28, 28) sample shape, which quietly made 2D toy data impossible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracking import NullTracker, Tracker
from utils import EMA
from viz import save_loss_curve


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 2e-3
    ema_decay: float = 0.999
    grad_clip: float | None = 1.0
    preview_every_epochs: int = 1
    log_every_steps: int = 10  # tracker cadence; local files are unaffected
    out_dir: str = "runs/exp"
    max_steps: int | None = None  # cap optimizer steps (smoke tests)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        device: torch.device,
        config: TrainConfig,
        preview_fn: Callable[[torch.nn.Module, Path, int], dict[str, Path] | None] | None = None,
        meta: dict[str, Any] | None = None,
        tracker: Tracker | None = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.preview_fn = preview_fn
        # `meta` is written into the checkpoint so sampling can rebuild the
        # exact path/target/model without you re-typing flags. Getting this
        # wrong used to produce silently garbage samples.
        self.meta = meta or {}
        # A tracker only mirrors what is already written to disk, so
        # everything below works identically with NullTracker.
        self.tracker = tracker or NullTracker()

        self.opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.ema = EMA(self.model, decay=config.ema_decay)
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.losses: list[float] = []
        self.global_step = 0

    def fit(self, dataloader: DataLoader) -> None:
        cfg = self.config
        stop = False
        for epoch in range(cfg.epochs):
            if stop:
                break
            pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{cfg.epochs}")
            for batch in pbar:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)

                loss = self.loss_fn(self.model, x)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self.ema.update(self.model)

                self.global_step += 1
                self.losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if self.global_step % cfg.log_every_steps == 0:
                    self.tracker.log_scalars(
                        self.global_step,
                        {"loss": loss.item(), "epoch": epoch + 1},
                    )

                if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                    stop = True
                    break

            if stop or (epoch + 1) % cfg.preview_every_epochs == 0:
                self._preview(epoch + 1)
            self.save(epoch + 1)

    def _preview(self, epoch: int) -> None:
        if self.preview_fn is None:
            return
        self.model.eval()
        # Preview from the EMA weights -- they are what you would ship,
        # and early in training they look markedly better than the raw ones.
        # preview_fn returns {panel name: written file}, which is all the
        # tracker needs; the files exist on disk either way.
        produced = self.preview_fn(self.ema.module, self.out_dir, epoch)
        if produced:
            self.tracker.log_images(self.global_step, produced)
        self.model.train()

    def save(self, epoch: int) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.module.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "meta": self.meta,
                "config": asdict(self.config),
            },
            self.out_dir / "checkpoint.pt",
        )
        # Loss history was previously collected and thrown away, leaving
        # sample grids as the only way to compare two runs.
        (self.out_dir / "losses.json").write_text(json.dumps(self.losses))
        if self.losses:
            save_loss_curve(self.losses, self.out_dir / "loss_curve.png")

    def finish(self) -> None:
        self.tracker.finish()
