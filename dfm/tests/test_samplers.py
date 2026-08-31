"""Tests for the sampler axis, against velocity fields with known
closed-form solutions -- so a broken integrator fails numerically
rather than just producing bad-looking samples.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from paths import LinearPath
from samplers import SAMPLERS, euler, heun
from targets import VelocityTarget

PATH, TARGET = LinearPath(), VelocityTarget()
SHAPE = (32, 2)


class ConstantField(nn.Module):
    """v(x, t) = c. Exact solution: x(1) = x(0) + c."""

    def __init__(self, c: float = 1.5):
        super().__init__()
        self.c = c

    def forward(self, x, t):
        return torch.full_like(x, self.c)


class TimeRampField(nn.Module):
    """v(x, t) = t. Exact solution: x(1) = x(0) + 1/2."""

    def forward(self, x, t):
        return t.reshape(-1, *([1] * (x.dim() - 1))).expand_as(x)


def _start_point(shape):
    """Reproduce the x(0) a sampler draws, given the same seed."""
    torch.manual_seed(0)
    return torch.randn(shape)


@pytest.mark.parametrize("name,sampler", sorted(SAMPLERS.items()))
def test_shape_and_finiteness(name, sampler):
    torch.manual_seed(0)
    out = sampler(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                  steps=5, progress=False)
    assert out.shape == SHAPE
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name,sampler", sorted(SAMPLERS.items()))
def test_constant_field_is_exact(name, sampler):
    """Both solvers integrate a constant field exactly, at any step count."""
    c = 1.5
    torch.manual_seed(0)
    out = sampler(ConstantField(c), PATH, TARGET, SHAPE, torch.device("cpu"),
                  steps=3, progress=False)
    assert torch.allclose(out, _start_point(SHAPE) + c, atol=1e-5)


def test_heun_is_second_order_where_euler_is_not():
    """On v(x,t)=t the exact displacement is 1/2.

    Heun's trapezoid rule is exact for a field linear in t; Euler is not,
    and undershoots by exactly 1/(2n). This is the whole reason the
    sampler is a separate axis -- swapping it changes the answer on a
    fixed model.
    """
    steps = 4
    start = _start_point(SHAPE)

    torch.manual_seed(0)
    e = euler(TimeRampField(), PATH, TARGET, SHAPE, torch.device("cpu"),
              steps=steps, progress=False)
    torch.manual_seed(0)
    h = heun(TimeRampField(), PATH, TARGET, SHAPE, torch.device("cpu"),
             steps=steps, progress=False)

    assert torch.allclose(h, start + 0.5, atol=1e-5)                       # exact
    assert torch.allclose(e, start + (steps - 1) / (2 * steps), atol=1e-5)  # 3/8, not 1/2
    assert (h - (start + 0.5)).abs().max() < (e - (start + 0.5)).abs().max()


def test_trajectory_shape():
    steps = 6
    torch.manual_seed(0)
    traj = euler(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                 steps=steps, progress=False, return_trajectory=True)
    assert traj.shape == (steps + 1, *SHAPE)
    assert torch.allclose(traj[0], _start_point(SHAPE), atol=1e-6)
