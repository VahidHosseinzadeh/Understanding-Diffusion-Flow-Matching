"""The training objective, written once against (path, target).

    loss = E_{x_data, x_noise, t} [ w(t) * || f_theta(x_t, t) - y ||^2 ]

with x_t from the path, y from the target, and w(t) a weighting. The
loss does not know it is doing flow matching -- give it a different
path or target and it trains that instead. That is the payoff of the
three-axis split: this file never changes.

Two knobs live here rather than on the path or target, because they are
properties of how you *train*, not of the process itself:

  - the distribution t is drawn from
  - the per-timestep weighting w(t) -- which also decides *which space*
    the error is measured in, since moving a loss from one target's
    space to another's only ever rescales it by a function of t
    (`LossSpaceWeighting`)

Both matter more than they look. A uniform t spends equal effort on
every noise level; real runs often do better concentrating on the
middle, where the regression problem is hardest.
"""
from __future__ import annotations

from typing import Callable

import torch

from paths import Path
from targets import Target


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


# A weighting is handed the path and target the loss is using, not just t:
# the interesting ones are functions of alpha(t), beta(t) and of what the
# network predicts. Passing them in, rather than letting a closure capture
# its own copies, means the weighting and the loss can never disagree.
Weighting = Callable[[Path, Target, torch.Tensor], torch.Tensor]


class LossSpaceWeighting:
    """Measure the error in `space`, whatever the network predicts.

    What the network outputs and where its error is measured are separate
    choices. JiT (Li & He 2025, "Back to Basics: Let Denoising Generative
    Models Denoise") tabulates all nine pairs and trains x_data-prediction
    with a velocity loss. Moving an error between spaces only rescales it
    (`Target.velocity_error_scale`), so the loss in space S of a network
    predicting P is its plain loss times

        w(t) = (s_P(t) / s_S(t))^2

    and the prediction never needs converting. Mind the direction if you
    check this against a conversion table: its (P, S) entry -- the error
    in S per unit error in P -- is s_P / s_S, but if the table is instead
    read as one *denominator* per space (beta for x_data, alpha for noise,
    det for velocity, so |d| = |det| / |s|), the same weight reads
    (d_S / d_P)^2. Reciprocal bookkeeping, identical numbers. The part
    that is not a convention: x_data measured in velocity must blow *up*
    as t -> 1, where beta -> 0. On `LinearPath`:

        x_data measured in velocity   1/(1-t)^2
        noise  measured in velocity   1/t^2
        x_data measured in noise      t^2/(1-t)^2 = SNR(t), the identity
                                      behind Kingma et al. 2021 (VDM)

    The weight diverges wherever the conversion is singular. Uncapped,
    x_data-in-velocity has E[w] = infinity under uniform t, and a sample
    drawn at t = 0.9999 carries 10^8 times the weight of one at t = 0;
    `max_weight` caps it. JiT's code converts both x_pred and x_data to
    velocity, dividing each by (1-t).clamp_min(0.05). The x_t terms cancel
    in the difference, so that loss is exactly this weighting at the
    default cap, 400 = 1/0.05^2. Capping x_data-in-noise the same way is
    Min-SNR-gamma (Hang et al. 2023).

    The loss comes out in S's units, so runs measured in the same space
    have comparable loss curves (up to the cap) even when their targets
    differ -- plain losses of different targets are not.
    """

    def __init__(self, space: Target, max_weight: float = 400.0):
        self.space = space
        self.max_weight = max_weight

    def __call__(self, path: Path, target: Target, t: torch.Tensor) -> torch.Tensor:
        ratio = target.velocity_error_scale(path, t) / self.space.velocity_error_scale(path, t)
        return ratio.pow(2).clamp(max=self.max_weight)

    def __repr__(self) -> str:
        return f"LossSpaceWeighting(space={self.space!r}, max_weight={self.max_weight:g})"


def interpolant_loss(
    model,
    x_data: torch.Tensor,
    path: Path,
    target: Target,
    t_sampler: Callable[[int, torch.device], torch.Tensor] = uniform_t,
    weighting: Weighting | None = None,
    return_per_sample: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One MSE step of the interpolant objective. Returns a scalar.

    The four lines that matter are the four in the middle: draw noise,
    draw a time, interpolate, regress. Everything a specific method adds
    on top of that lives behind `path` and `target`.

    `weighting(path, target, t)` returns a (B,) per-sample weight.
    Measuring the error in another target's space is one such weighting
    (`LossSpaceWeighting`), so it needs no branch here.

    With `return_per_sample=True` also returns the detached per-sample
    MSE and the t each sample was drawn at, which is what
    `diagnostics.LossTimeProfile` bins. The per-sample value is the
    *unweighted* error: the weighting is a statement about which samples
    should influence the gradient, not about how hard they actually are,
    and conflating the two would hide exactly the boundary effects the
    profile exists to find.
    """

    # sampling noise and time (note that we have independent coupling of noise and data here)
    batch = x_data.shape[0]
    x_noise = torch.randn_like(x_data)
    t = t_sampler(batch, x_data.device)


    # interpolation depending what is the path and defining x_t, and 
    # also definig the target our model wants to learn (can be velocity, score, noise, x_data, etc.)
    x_t = path.interpolate(x_data, x_noise, t)
    y = target.regression_target(path, x_data, x_noise, t)
    pred = model(x_t, t)

    # loss between prediction and the target, optionally weighted by a function of t
    se = (pred - y).pow(2).flatten(1).mean(dim=1)  # per-sample squared error
    per_sample = se.detach()
    if weighting is not None:
        se = se * weighting(path, target, t)
    loss = se.mean()
    if return_per_sample:
        return loss, per_sample, t.detach()
    return loss
