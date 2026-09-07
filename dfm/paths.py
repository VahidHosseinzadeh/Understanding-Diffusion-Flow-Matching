"""Axis 1 of 3: the **path** -- how data and noise are interpolated.

A path defines, for t in [0, 1], a Gaussian probability path

    p_t(x | z) = N(alpha(t) * z, beta(t)^2 * I)

i.e. a sample from it is drawn as

    x_t = alpha(t) * x_data + beta(t) * x_noise

together with the time derivatives alpha'(t) and beta'(t), which give
the velocity of a point travelling along that path:

    dx_t/dt = alpha'(t) * x_data + beta'(t) * x_noise

That is the entire definition. Every generative process in this family
-- rectified flow, DDPM/VP diffusion, sub-VP, EDM's variance-exploding
schedule -- is one choice of (alpha, beta). This is the decomposition
from Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models": separate the schedule from what the network predicts
from how you integrate, and the differences between methods become a
table rather than a pile of separate codebases.

NOTATION -- following Holderrieth & Erives, "An Introduction to Flow
Matching and Diffusion Models" (arXiv 2506.02070), our reference text:
    alpha(t), beta(t)   defined here. The two coefficients of the
                        Gaussian path p_t(x|z) = N(alpha(t) z, beta(t)^2 I).
    sigma, sigma_t      deliberately NOT used here. Reserved for the
                        unrelated SDE diffusion coefficient some
                        samplers add on top of a learned path (the "SDE
                        extension trick"):
                            dX_t = [u_t(X_t) + (sigma_t^2/2) score_t(X_t)] dt + sigma_t dW_t
                        We have no code for that yet. If you add it,
                        keep it out of this file -- beta(t) is a path
                        parameter fixed before training, sigma_t is a
                        sampling-time knob on a fixed, trained model;
                        reusing one bare letter for both is exactly the
                        kind of ambiguity that costs an hour of
                        debugging when you're rereading your own code.

TIME CONVENTION
    t = 0 is NOISE, t = 1 is DATA.
    So alpha(0) = 0, beta(0) = 1 and alpha(1) = 1, beta(1) = 0.
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
        beta(t) = sqrt(1 - abar(t))
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
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient on the noise endpoint. beta(0)=1, beta(1)=0."""

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d alpha / dt."""

    @abstractmethod
    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d beta / dt."""

    # -- derived quantities: these work for any path ---------------------

    def interpolate(self, x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = alpha(t) * x_data + beta(t) * x_noise."""
        alpha_t = expand_to(self.alpha(t), x_data)
        beta_t = expand_to(self.beta(t), x_data)
        return alpha_t * x_data + beta_t * x_noise

    def velocity(self, x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """dx_t/dt = alpha'(t) * x_data + beta'(t) * x_noise.

        Note this is the velocity of the *conditional* path joining one
        specific (x_noise, x_data) pair. The network learns its
        conditional expectation, which is the marginal velocity field --
        that swap is the whole content of the flow matching theorem.
        """
        alpha_dot_t = expand_to(self.alpha_dot(t), x_data)
        beta_dot_t = expand_to(self.beta_dot(t), x_data)
        return alpha_dot_t * x_data + beta_dot_t * x_noise

    def solve(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover (x_data, x_noise) from a point and its velocity.

        x_t and v are two linear equations in the two unknowns:
            x_t = alpha   * x_data + beta   * x_noise
            v   = alpha'  * x_data + beta'  * x_noise
        with determinant D = alpha * beta' - beta * alpha', giving
            x_data  = ( beta' * x_t - beta * v ) / D
            x_noise = ( alpha * v   - alpha' * x_t ) / D

        This is what makes the *target* axis cheap: any one of
        {velocity, x_data, x_noise} determines the other two, so
        switching parameterisation never requires a new path or sampler.
        """
        alpha_t = expand_to(self.alpha(t), x_t)
        beta_t = expand_to(self.beta(t), x_t)
        alpha_dot_t = expand_to(self.alpha_dot(t), x_t)
        beta_dot_t = expand_to(self.beta_dot(t), x_t)
        det = alpha_t * beta_dot_t - beta_t * alpha_dot_t
        x_data = (beta_dot_t * x_t - beta_t * v) / det
        x_noise = (alpha_t * v - alpha_dot_t * x_t) / det
        return x_data, x_noise


class LinearPath(Path):
    """Straight-line path -- rectified flow (Liu et al. 2022) and the
    conditional flow matching of Lipman et al. 2023.

        x_t = t * x_data + (1 - (1 - beta_min) * t) * x_noise

    The velocity is constant along each conditional path:
        dx_t/dt = x_data - (1 - beta_min) * x_noise
    which is why flow matching can sample accurately in few steps: the
    trajectories it has to integrate are as close to straight as this
    family gets.

    beta_min > 0 leaves a little noise at the data endpoint (some
    conditional-flow-matching variants do this for numerical headroom);
    beta_min = 0 is plain rectified flow and the default.
    """

    def __init__(self, beta_min: float = 0.0):
        self.beta_min = beta_min

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - self.beta_min) * t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, -(1.0 - self.beta_min))

    def __repr__(self) -> str:
        return f"LinearPath(beta_min={self.beta_min})"


PATHS = {"linear": LinearPath}
