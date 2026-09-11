"""High-dimensional diagnostics -- the numbers, not the pictures.

Sample grids tell you whether a model works. They do not tell you *where*
it is failing, and in high dimensions there is no velocity-field plot to
fall back on. These four probe the places this family of methods
characteristically breaks:

  loss_vs_time            is training uniformly hard across t, or is some
                          band of noise levels dominating the objective?
  velocity_norm_vs_time   does the drift explode near an endpoint? This is
                          the failure mode non-velocity targets have by
                          construction -- see `Target.t_range`.
  sampler budget matrix   at a fixed number of network calls, which solver
                          actually wins?

Everything here returns plain tensors/arrays. Rendering lives in viz.py,
logging in tracking.py.
"""
from __future__ import annotations

from typing import Callable

import torch

from paths import Path
from samplers import NFE_PER_STEP, SAMPLERS
from targets import Target


class LossTimeProfile:
    """Running mean of the per-sample loss, bucketed by t.

    Fed from inside the training loss, so it costs one scatter-add per
    step and needs no extra forward passes. A healthy profile is roughly
    flat; a spike at one end means that band of noise levels is soaking
    up the gradient budget, which is the usual signature of an
    ill-conditioned parameterisation or a schedule with too much mass
    near an endpoint.
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.reset()

    def reset(self) -> None:
        self._sum = torch.zeros(self.n_bins, dtype=torch.float64)
        self._count = torch.zeros(self.n_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, t: torch.Tensor, per_sample_loss: torch.Tensor) -> None:
        """`t` and `per_sample_loss` are both (B,), detached."""
        # .cpu() before .to(float64): MPS has no float64, so casting on
        # device raises rather than falling back.
        t = t.detach().cpu().to(torch.float64).clamp(0.0, 1.0)
        loss = per_sample_loss.detach().cpu().to(torch.float64)
        # clamp so t == 1.0 lands in the last bin rather than out of range
        idx = (t * self.n_bins).long().clamp(max=self.n_bins - 1)
        self._sum.scatter_add_(0, idx, loss)
        self._count.scatter_add_(0, idx, torch.ones_like(loss))

    def summary(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(bin_centres, mean_loss_per_bin, count_per_bin).

        Empty bins come back as NaN rather than 0 -- a bin nothing landed
        in is missing data, and plotting it as zero would invent a dip.
        """
        centres = (torch.arange(self.n_bins, dtype=torch.float64) + 0.5) / self.n_bins
        means = torch.where(self._count > 0, self._sum / self._count.clamp(min=1),
                            torch.full_like(self._sum, float("nan")))
        return centres, means, self._count.clone()

    def is_empty(self) -> bool:
        return bool(self._count.sum() == 0)


@torch.no_grad()
def velocity_norm_profile(
    model,
    path: Path,
    target: Target,
    x_data: torch.Tensor,
    n_times: int = 40,
    t_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """||u_theta(x_t, t)||_2 as a function of t. Returns (times, mean, max).

    Probes the *converted* velocity -- whatever the network predicts, run
    through `Target.to_velocity` -- because that conversion is where noise-
    and score-prediction blow up, not the raw network output. A profile
    that shoots up at one end is the boundary explosion; compare against
    a velocity-target run, which should stay flat.

    Deliberately sweeps the full [0, 1] by default rather than the
    target's safe `t_range`, since seeing the explosion is the point.
    """
    lo, hi = t_range if t_range is not None else (0.0, 1.0)
    device = x_data.device
    times = torch.linspace(lo, hi, n_times, device=device)
    means, maxes = [], []

    for t_scalar in times:
        t = torch.full((x_data.shape[0],), t_scalar.item(), device=device)
        x_noise = torch.randn_like(x_data)
        x_t = path.interpolate(x_data, x_noise, t)
        v = target.to_velocity(path, x_t, t, model(x_t, t))
        norms = v.flatten(1).norm(dim=1)
        means.append(norms.mean())
        maxes.append(norms.max())

    return times.cpu(), torch.stack(means).cpu(), torch.stack(maxes).cpu()


@torch.no_grad()
def sampler_budget_matrix(
    model,
    path: Path,
    target: Target,
    shape: tuple[int, ...],
    device: torch.device,
    budgets: tuple[int, ...] = (10, 20),
    sampler_names: tuple[str, ...] = ("euler", "heun", "euler_maruyama"),
    seed: int = 0,
) -> dict[tuple[str, int], torch.Tensor]:
    """Sample every (solver, budget) pair at *equal network calls*.

    Budgets are counted in model evaluations, not steps: Heun calls the
    field twice per step, so 20 NFE means 20 Euler steps but only 10 Heun
    steps. Comparing at equal steps quietly hands Heun double the compute
    and is the most common way this comparison gets reported wrongly.

    Every cell starts from the same noise (same seed), so differences are
    the solver's doing and not a different draw.
    """
    out: dict[tuple[str, int], torch.Tensor] = {}
    for name in sampler_names:
        per_step = NFE_PER_STEP[name]
        for nfe in budgets:
            steps = max(1, nfe // per_step)
            torch.manual_seed(seed)
            out[(name, nfe)] = SAMPLERS[name](
                model, path, target, shape, device, steps=steps, progress=False
            )
    return out
