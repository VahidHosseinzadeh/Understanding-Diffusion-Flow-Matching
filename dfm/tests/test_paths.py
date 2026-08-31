"""Tests for the path axis.

These check the *math*, not just shapes: a path that returns
inconsistent alpha_dot/sigma_dot would train a subtly wrong velocity
field and still produce plausible-looking pictures, so the derivative
check below is the one that actually protects you.
"""
from __future__ import annotations

import pytest
import torch

from paths import LinearPath

ALL_PATHS = [LinearPath(sigma_min=0.0), LinearPath(sigma_min=0.01)]


@pytest.mark.parametrize("path", ALL_PATHS)
def test_endpoints(path):
    """t=0 must be pure noise, t=1 pure data (up to sigma_min)."""
    x_data, x_noise = torch.randn(8, 2), torch.randn(8, 2)

    at_0 = path.interpolate(x_data, x_noise, torch.zeros(8))
    assert torch.allclose(at_0, x_noise, atol=1e-6)

    at_1 = path.interpolate(x_data, x_noise, torch.ones(8))
    assert torch.allclose(at_1, x_data + path.sigma_min * x_noise, atol=1e-6)


@pytest.mark.parametrize("path", ALL_PATHS)
def test_velocity_matches_finite_difference(path):
    """velocity() must be the actual time derivative of interpolate().

    Central difference: (x_{t+h} - x_{t-h}) / 2h -> dx/dt as h -> 0.
    This catches an alpha_dot/sigma_dot that disagrees with its own
    alpha/sigma, which is the classic error when adding a new schedule.
    """
    # float64: a central difference with h=1e-4 cancels away roughly four
    # significant digits, which is most of what float32 has.
    x_data = torch.randn(16, 3, dtype=torch.float64)
    x_noise = torch.randn(16, 3, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64) * 0.8 + 0.1  # keep off the endpoints
    h = 1e-6

    fd = (
        path.interpolate(x_data, x_noise, t + h)
        - path.interpolate(x_data, x_noise, t - h)
    ) / (2 * h)
    assert torch.allclose(fd, path.velocity(x_data, x_noise, t), atol=1e-8)


@pytest.mark.parametrize("path", ALL_PATHS)
def test_solve_round_trips(path):
    """(x_t, v) -> (x_data, x_noise) must invert the interpolation.

    This identity is what lets the target axis exist: any prediction
    can be converted into any other parameterisation.
    """
    x_data, x_noise = torch.randn(16, 2), torch.randn(16, 2)
    t = torch.rand(16)

    x_t = path.interpolate(x_data, x_noise, t)
    v = path.velocity(x_data, x_noise, t)
    rec_data, rec_noise = path.solve(x_t, v, t)

    assert torch.allclose(rec_data, x_data, atol=1e-4)
    assert torch.allclose(rec_noise, x_noise, atol=1e-4)


def test_linear_path_velocity_is_constant_in_t():
    """Rectified flow's defining property: along one conditional path
    the velocity does not depend on t. This is why it can take big steps."""
    path = LinearPath()
    x_data, x_noise = torch.randn(8, 2), torch.randn(8, 2)

    v_early = path.velocity(x_data, x_noise, torch.full((8,), 0.1))
    v_late = path.velocity(x_data, x_noise, torch.full((8,), 0.9))

    assert torch.allclose(v_early, v_late, atol=1e-6)
    assert torch.allclose(v_early, x_data - x_noise, atol=1e-6)


def test_broadcasting_over_image_shapes():
    """Paths must work unchanged on (B, C, H, W), not just (B, D)."""
    path = LinearPath()
    x_data, x_noise = torch.randn(4, 1, 28, 28), torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    assert path.interpolate(x_data, x_noise, t).shape == x_data.shape
    assert path.velocity(x_data, x_noise, t).shape == x_data.shape
