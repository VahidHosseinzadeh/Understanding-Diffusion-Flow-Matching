"""Axis 2 of 3: the **target** -- what the network is asked to predict.

A path says where x_t lives. It does not say what the network should
output there. Any of

    the velocity   dx_t/dt
    the data point x_data      ("x0-prediction" in DDPM terms)
    the noise      x_noise     ("eps-prediction")

determines the other two via `Path.solve`, so they parameterise the
*same* underlying model. They are not equivalent in practice: they put
the regression difficulty in different places and weight timesteps
differently, which is why the choice matters empirically even though
it is a no-op mathematically.

A Target therefore needs exactly two methods:

    regression_target(...)  what to put on the right-hand side of the MSE
    to_velocity(...)        how the sampler turns a prediction back into
                            dx/dt, since every sampler here integrates
                            an ODE in velocity

Keeping `to_velocity` on this axis is what lets samplers stay ignorant
of parameterisation: `samplers.euler` works with any target you add.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from paths import Path, expand_to


class Target(ABC):
    @abstractmethod
    def regression_target(
        self, path: Path, x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """The tensor the network should match at (x_t, t)."""

    @abstractmethod
    def to_velocity(
        self, path: Path, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        """Convert a raw network output into dx/dt for the sampler."""

    # Not abstract: a target valid on all of [0, 1] need not think about this.
    def t_range(self) -> tuple[float, float]:
        """The sub-interval of [0, 1] a sampler may evaluate this target on.

        `to_velocity` inverts the interpolant for every parameterisation
        except velocity, and that division blows up wherever alpha(t) or
        beta(t) vanishes. Rather than teach each sampler which targets
        are singular where -- a branch on target type, exactly what this
        axis split exists to prevent -- the target states its own domain
        and samplers integrate over whatever they are given.
        """
        return 0.0, 1.0


class VelocityTarget(Target):
    """Predict the velocity directly. This is standard flow matching.

    The conversion for the sampler is the identity, which is the reason
    flow matching reads so much more simply than diffusion: there is no
    algebra between what the network outputs and what the ODE solver
    consumes.
    """

    def regression_target(self, path, x_data, x_noise, t):
        return path.velocity(x_data, x_noise, t)

    def to_velocity(self, path, x_t, t, pred):
        return pred

    def __repr__(self) -> str:
        return "VelocityTarget()"


# Inverting the interpolant means dividing by alpha(t) or beta(t), and
# each vanishes at one endpoint: alpha(0) = 0, beta(1) = 0. The division
# is then 0/0 -- indeterminate, not infinite, because the numerator
# vanishes too. We floor the magnitude of the denominator so it returns
# something finite rather than NaN, but "finite" is not "accurate": near
# its bad endpoint each of these parameterisations amplifies whatever
# error the network has. That is a real property of the choice, not a
# bug in the algebra, and it is half of what EDM's preconditioning
# exists to fix. VelocityTarget has no such endpoint because it never
# inverts anything.
_EPS = 1e-4

# How far a sampler must stay from the endpoint where each parameterisation
# is singular. These differ because the conditioning does: measured on the
# 2D moons task, DataTarget is already usable at 1e-2, while NoiseTarget
# needs ~5e-2 before the error it amplifies stops dominating the samples.
# Trimming costs a little accuracy of its own (integration no longer covers
# the full interval), so bigger is not better -- these are the smallest
# values that worked.
_DATA_MARGIN = 1e-2
_NOISE_MARGIN = 5e-2


def _floor_magnitude(c: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Keep |c| >= eps for use as a denominator, preserving c's sign."""
    sign = torch.where(c < 0, -1.0, 1.0)
    return sign * c.abs().clamp(min=eps)


class DataTarget(Target):
    """Predict the data endpoint, x_data.

    Called "x0-prediction" in the DDPM literature, where the data
    endpoint carries index 0. Nothing here is indexed that way -- t=1 is
    data -- so the target is named for what it predicts.

    Well conditioned near t=0 (mostly noise, so recovering the noise
    endpoint from a predicted x_data is easy) and badly conditioned near
    t=1, where beta(t) -> 0.
    """

    def regression_target(self, path, x_data, x_noise, t):
        return x_data

    def to_velocity(self, path, x_t, t, pred):
        # pred is x_data; recover the other endpoint from the interpolant
        #   x_t = alpha*x_data + beta*x_noise  =>  x_noise = (x_t - alpha*pred)/beta
        alpha_t = expand_to(path.alpha(t), x_t)
        beta_t = expand_to(path.beta(t), x_t)
        alpha_dot_t = expand_to(path.alpha_dot(t), x_t)
        beta_dot_t = expand_to(path.beta_dot(t), x_t)
        x_noise = (x_t - alpha_t * pred) / _floor_magnitude(beta_t)
        return alpha_dot_t * pred + beta_dot_t * x_noise

    def t_range(self) -> tuple[float, float]:
        return 0.0, 1.0 - _DATA_MARGIN

    def __repr__(self) -> str:
        return "DataTarget()"


class NoiseTarget(Target):
    """Predict the noise endpoint, x_noise.

    This is DDPM's eps-prediction, and the most common parameterisation
    in the diffusion literature by a wide margin.

    Its conditioning is the mirror image of DataTarget's: fine near t=1,
    bad near t=0 where alpha(t) -> 0. Note that DDPM never feels this,
    because its samplers are written directly in terms of eps and never
    convert to a velocity -- the singularity is a cost of routing
    everything through one ODE interface, which is the trade this
    package makes to keep the three axes independent.
    """

    def regression_target(self, path, x_data, x_noise, t):
        return x_noise

    def to_velocity(self, path, x_t, t, pred):
        # pred is x_noise;  x_data = (x_t - beta*pred)/alpha
        alpha_t = expand_to(path.alpha(t), x_t)
        beta_t = expand_to(path.beta(t), x_t)
        alpha_dot_t = expand_to(path.alpha_dot(t), x_t)
        beta_dot_t = expand_to(path.beta_dot(t), x_t)
        x_data = (x_t - beta_t * pred) / _floor_magnitude(alpha_t)
        return alpha_dot_t * x_data + beta_dot_t * pred

    def t_range(self) -> tuple[float, float]:
        return _NOISE_MARGIN, 1.0

    def __repr__(self) -> str:
        return "NoiseTarget()"


# NEXT EXERCISE -- score prediction.
#
#   score_t(x_t) = grad_x log p_t(x_t),  which for a Gaussian path is
#   just the noise endpoint rescaled:
#       score = -x_noise / beta(t)
#   so ScoreTarget.regression_target returns that, and to_velocity
#   recovers x_noise = -beta(t) * pred and then reuses NoiseTarget's two
#   lines. Score- and noise-prediction are the same object up to a
#   t-dependent factor, which is the bridge between score-based models
#   and DDPM.


TARGETS = {
    "velocity": VelocityTarget,
    "x_data": DataTarget,
    "noise": NoiseTarget,
}
