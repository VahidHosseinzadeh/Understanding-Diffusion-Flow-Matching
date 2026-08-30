"""Generic training loop shared by every process.

Trainer only knows about the small interface a process must expose
(`training_loss`, `sample`) -- it has no idea whether it's driving
DDPM or flow matching, which is the whole point: adding a third
process later (say, a consistency model) means writing a new class
with that interface, not touching this file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import EMA, save_image_grid


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 2e-4
    ema_decay: float = 0.999
    grad_clip: float | None = 1.0
    sample_every_epochs: int = 1
    sample_batch: int = 64
    out_dir: str = "runs/exp"
    log_every_steps: int = 50
    max_steps: int | None = None  # cap total optimizer steps (useful for smoke tests)


class Trainer:
    def __init__(self, model: torch.nn.Module, process, device: torch.device, config: TrainConfig):
        self.model = model.to(device)
        self.process = process
        self.device = device
        self.config = config
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
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
            for x, _ in pbar:
                x = x.to(self.device)
                loss = self.process.training_loss(self.model, x)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self.ema.update(self.model)

                self.global_step += 1
                self.losses.append(loss.item())
                if self.global_step % cfg.log_every_steps == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                    stop = True
                    break

            if (epoch + 1) % cfg.sample_every_epochs == 0 or stop:
                self._sample_and_save(epoch + 1)
            self._save_checkpoint(epoch + 1)

    @torch.no_grad()
    def _sample_and_save(self, epoch: int) -> None:
        self.model.eval()
        shape = (self.config.sample_batch, 1, 28, 28)
        samples = self.process.sample(self.ema.module, shape, self.device, progress=False)
        save_image_grid(samples, self.out_dir / f"samples_epoch{epoch:04d}.png", nrow=8)
        self.model.train()

    def _save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.module.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
            },
            self.out_dir / "checkpoint.pt",
        )
