"""Axis 1 of 3: the **path** -- how data and noise are interpolated.

A path defines, for t in [0, 1], a Gaussian interpolant

    x_t = alpha(t) * x_data + sigma(t) * x_noise

together with the time derivatives alpha'(t) and sigma'(t), which give
the velocity of a point travelling along that path:

    dx_t/dt = alpha'(t) * x_data + sigma'(t) * x_noise

That is the entire definition. Every generative process in this family
-- rectified flow, DDPM/VP diffusion, sub-VP, EDM's variance-exploding
schedule -- is one choice of (alpha, sigma). This is the decomposition
from Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models": separate the schedule from what the network predicts
from how you integrate, and the differences between methods become a
table rather than a pile of separate codebases.

TIME CONVENTION
    t = 0 is NOISE, t = 1 is DATA.
    So alpha(0) = 0, sigma(0) = 1 and alpha(1) = 1, sigma(1) = 0.
    Sampling therefore integrates *forward* in time, 0 -> 1.

    The DDPM literature runs the opposite way (t=0 is data, t=T is
    noise). When you add a variance-preserving path later, flip its
    schedule to match this convention rather than special-casing the
    samplers -- that is exactly the kind of branching this layout
    exists to avoid.

ADDING A PATH
    Subclass `Path` and implement four scalar functions of t. You get
    `interpolate`, `velocity` and `solve` for free, and every target,
    sampler and loss in the package works with it unchanged.

    Variance-preserving (DDPM) would be, with abar the usual cumulative
    product reparameterised so t=1 is data:
        alpha(t) = sqrt(abar(t))
        sigma(t) = sqrt(1 - abar(t))
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def expand_to(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample scalar (B,) so it broadcasts against x (B, ...)."""
    return c.reshape(-1, *([1] * (x.dim() - 1)))


class Path(ABC):
    """A Gaussian interpolant between noise (t=0) and data (t=1)."""

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient on the data endpoint. alpha(0)=0, alpha(1)=1."""

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient on the noise endpoint. sigma(0)=1, sigma(1)=0."""

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d alpha / dt."""

    @abstractmethod
    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d sigma / dt."""

    # -- derived quantities: these work for any path ---------------------

    def interpolate(self, x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = alpha(t) * x_data + sigma(t) * x_noise."""
        a = expand_to(self.alpha(t), x_data)
        s = expand_to(self.sigma(t), x_data)
        return a * x_data + s * x_noise

    def velocity(self, x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """dx_t/dt = alpha'(t) * x_data + sigma'(t) * x_noise.

        Note this is the velocity of the *conditional* path joining one
        specific (x_noise, x_data) pair. The network learns its
        conditional expectation, which is the marginal velocity field --
        that swap is the whole content of the flow matching theorem.
        """
        a_dot = expand_to(self.alpha_dot(t), x_data)
        s_dot = expand_to(self.sigma_dot(t), x_data)
        return a_dot * x_data + s_dot * x_noise

    def solve(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover (x_data, x_noise) from a point and its velocity.

        x_t and v are two linear equations in the two unknowns:
            x_t = alpha   * x_data + sigma   * x_noise
            v   = alpha'  * x_data + sigma'  * x_noise
        with determinant D = alpha * sigma' - sigma * alpha', giving
            x_data  = ( sigma' * x_t - sigma * v ) / D
            x_noise = ( alpha  * v   - alpha' * x_t ) / D

        This is what makes the *target* axis cheap: any one of
        {velocity, x_data, x_noise} determines the other two, so
        switching parameterisation never requires a new path or sampler.
        """
        a = expand_to(self.alpha(t), x_t)
        s = expand_to(self.sigma(t), x_t)
        a_dot = expand_to(self.alpha_dot(t), x_t)
        s_dot = expand_to(self.sigma_dot(t), x_t)
        det = a * s_dot - s * a_dot
        x_data = (s_dot * x_t - s * v) / det
        x_noise = (a * v - a_dot * x_t) / det
        return x_data, x_noise


class LinearPath(Path):
    """Straight-line path -- rectified flow (Liu et al. 2022) and the
    conditional flow matching of Lipman et al. 2023.

        x_t = t * x_data + (1 - (1 - sigma_min) * t) * x_noise

    The velocity is constant along each conditional path:
        dx_t/dt = x_data - (1 - sigma_min) * x_noise
    which is why flow matching can sample accurately in few steps: the
    trajectories it has to integrate are as close to straight as this
    family gets.

    sigma_min > 0 leaves a little noise at the data endpoint (some
    conditional-flow-matching variants do this for numerical headroom);
    sigma_min = 0 is plain rectified flow and the default.
    """

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - self.sigma_min) * t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, -(1.0 - self.sigma_min))

    def __repr__(self) -> str:
        return f"LinearPath(sigma_min={self.sigma_min})"


PATHS = {"linear": LinearPath}
