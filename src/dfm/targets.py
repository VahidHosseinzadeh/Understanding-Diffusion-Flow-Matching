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

from .paths import Path


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


# ---------------------------------------------------------------------
# NEXT EXERCISE -- the other two parameterisations.
#
# Both are about four lines each, because `Path.solve` already does the
# work. Implementing them lets you run the comparison that motivates
# this axis existing at all: identical path, identical data, identical
# sampler, three different regression targets.
#
#   class DataTarget(Target):          # x0-prediction
#       def regression_target(self, path, x_data, x_noise, t):
#           return x_data
#       def to_velocity(self, path, x_t, t, pred):
#           # pred is x_data; recover x_noise from the interpolant
#           #   x_noise = (x_t - alpha * pred) / sigma
#           # then    v = alpha' * pred + sigma' * x_noise
#
#   class NoiseTarget(Target):         # eps-prediction, as in DDPM
#       def regression_target(self, path, x_data, x_noise, t):
#           return x_noise
#       def to_velocity(self, path, x_t, t, pred):
#           #   x_data = (x_t - sigma * pred) / alpha
#           # then    v = alpha' * x_data + sigma' * pred
#
# Watch for the endpoint singularities when you do: DataTarget divides
# by sigma(1) = 0 and NoiseTarget divides by alpha(0) = 0. That is not a
# bug in the algebra, it is a real property of those parameterisations,
# and handling it is half of what EDM's preconditioning is for.
# ---------------------------------------------------------------------


TARGETS = {"velocity": VelocityTarget}
