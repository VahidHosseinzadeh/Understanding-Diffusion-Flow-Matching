"""Classifier-free guidance (Ho & Salimans 2022), as a wrapper around the network.

A class-conditional network trained with label dropout (`losses.drop_labels`)
holds two fields in one set of weights: the conditional prediction at a
label y and the unconditional one at NULL_LABEL. Guidance samples with

    guided = (1 - w) * uncond + w * cond = uncond + w * (cond - uncond)

so w = 1 is plain conditional sampling, w = 0 is unconditional, and w > 1
extrapolates away from the unconditional prediction. For a Gaussian path
the bracket is the gradient of an implicit classifier, which w amplifies.

Why a wrapper, and why combining raw predictions is enough: every
target's `to_velocity` is affine in the prediction, and the two weights
sum to one, so mixing predictions and then converting equals converting
and then mixing. Guidance therefore never needs to know the target, and
`ClassifierFreeGuidance` keeps the `forward(x, t)` contract -- every
sampler runs it unchanged.

It is not free: each guided evaluation runs the network on a doubled
batch, conditional and unconditional together. `network_passes` measures
what a whole trajectory costs.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from embeddings import NULL_LABEL
from paths import Path, expand_to
from targets import Target


class ClassifierFreeGuidance(nn.Module):
    """The guided prediction for fixed labels, as a model of (x, t).

    `interval = (lo, hi)` applies guidance only while lo <= t <= hi and
    samples plain conditionally elsewhere, as in Kynkaanniemi et al. 2024
    ("Applying guidance in a limited interval improves sample and
    distribution quality"). Recall that t = 0 is noise here. Outside the
    interval a step costs one network pass instead of two.

    `network_passes` counts batch-sized network evaluations made so far.
    """

    def __init__(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        scale: float,
        interval: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.model = model
        self.labels = labels
        self.scale = scale
        self.interval = interval
        self.network_passes = 0

    def _net(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, passes: int) -> torch.Tensor:
        self.network_passes += passes
        return self.model(x, t, y)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y = self.labels.to(x.device)
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"{y.shape[0]} labels for a batch of {x.shape[0]}")
        lo, hi = self.interval
        active = (t >= lo) & (t <= hi)

        if self.scale == 1.0 or not active.any():
            return self._net(x, t, y, passes=1)
        null = torch.full_like(y, NULL_LABEL)
        if self.scale == 0.0 and active.all():
            return self._net(x, t, null, passes=1)

        both = self._net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]), passes=2)
        cond, uncond = both.chunk(2)
        guided = uncond + self.scale * (cond - uncond)
        return torch.where(expand_to(active, guided), guided, cond)


def balanced_labels(n: int, num_classes: int) -> torch.Tensor:
    """0, 1, ..., C-1, 0, 1, ...: equal class counts for scoring, and in a
    grid with `nrow=num_classes` every column is one class."""
    return torch.arange(n) % num_classes


class _Stub(nn.Module):
    def forward(self, x, t, y=None):
        return torch.zeros_like(x)


@torch.no_grad()
def network_passes(
    sampler: Callable,
    path: Path,
    target: Target,
    sample_shape: tuple[int, ...],
    steps: int,
    scale: float,
    interval: tuple[float, float] = (0.0, 1.0),
) -> int:
    """Batch-sized network passes one guided trajectory costs.

    Measured by running the sampler on a stand-in network, not computed:
    which steps fall inside the guidance interval depends on the sampler's
    time grid, which is the sampler's business alone.
    """
    probe = ClassifierFreeGuidance(_Stub(), torch.zeros(1, dtype=torch.long), scale, interval)
    with torch.random.fork_rng(devices=[]):  # stochastic samplers must not shift the caller's RNG
        sampler(probe, path, target, (1, *sample_shape), torch.device("cpu"),
                steps=steps, progress=False)
    return probe.network_passes
