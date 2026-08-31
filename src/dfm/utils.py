"""Small shared utilities: seeding, device selection, EMA.

Plotting lives in dfm.viz."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick a device. 'auto' prefers CUDA, then Apple MPS, then CPU.

    On the cluster this will pick up CUDA automatically; on a Mac
    running natively (not in a Linux VM) it would pick up MPS; here
    it will fall back to CPU.
    """
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of model parameters.

    Standard trick in diffusion/flow-matching training: sample from the
    EMA weights instead of the raw weights for noticeably cleaner
    generations. Call `update()` after every optimizer step and use
    `ema.module` (or the `swap_in`/`swap_out` context) at sampling time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            model_v = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(model_v, alpha=1 - self.decay)
            else:
                v.copy_(model_v)
