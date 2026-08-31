"""The training objective, written once against (path, target).

    loss = E_{x_data, x_noise, t} [ w(t) * || f_theta(x_t, t) - y ||^2 ]

with x_t from the path, y from the target, and w(t) a weighting. The
loss does not know it is doing flow matching -- give it a different
path or target and it trains that instead. That is the payoff of the
three-axis split: this file never changes.

Two knobs live here rather than on the path or target, because they are
properties of how you *train*, not of the process itself:

  - the distribution t is drawn from
  - the per-timestep weighting w(t)

Both matter more than they look. A uniform t spends equal effort on
every noise level; real runs often do better concentrating on the
middle, where the regression problem is hardest.
"""
from __future__ import annotations

from typing import Callable

import torch

from .paths import Path
from .targets import Target


def uniform_t(batch: int, device: torch.device) -> torch.Tensor:
    """t ~ U[0, 1]. The default, and what rectified flow assumes."""
    return torch.rand(batch, device=device)


def logit_normal_t(batch: int, device: torch.device, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """t = sigmoid(z), z ~ N(mean, std) -- the Stable Diffusion 3 choice.

    Concentrates samples near t = 0.5 and puts little weight on the
    endpoints, where the conditional velocity is easiest to predict and
    the gradient signal is least useful.
    """
    return torch.sigmoid(torch.randn(batch, device=device) * std + mean)


T_SAMPLERS = {"uniform": uniform_t, "logit_normal": logit_normal_t}


def interpolant_loss(
    model,
    x_data: torch.Tensor,
    path: Path,
    target: Target,
    t_sampler: Callable[[int, torch.device], torch.Tensor] = uniform_t,
    weighting: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """One MSE step of the interpolant objective. Returns a scalar.

    The four lines that matter are the four in the middle: draw noise,
    draw a time, interpolate, regress. Everything a specific method adds
    on top of that lives behind `path` and `target`.
    """
    batch = x_data.shape[0]
    x_noise = torch.randn_like(x_data)
    t = t_sampler(batch, x_data.device)

    x_t = path.interpolate(x_data, x_noise, t)
    y = target.regression_target(path, x_data, x_noise, t)
    pred = model(x_t, t)

    se = (pred - y).pow(2).flatten(1).mean(dim=1)  # per-sample squared error
    if weighting is not None:
        se = se * weighting(t)
    return se.mean()
