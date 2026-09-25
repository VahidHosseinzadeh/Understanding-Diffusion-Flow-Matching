"""Tests for the sampler axis, against velocity fields with known
closed-form solutions -- so a broken integrator fails numerically
rather than just producing bad-looking samples.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from paths import LinearPath
from samplers import SAMPLERS, euler, heun, network_calls, steps_for_budget
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


# euler_maruyama is stochastic by construction, so exactness applies only to
# the deterministic solvers. It gets its own tests below.
DETERMINISTIC = {k: v for k, v in SAMPLERS.items() if k != "euler_maruyama"}


@pytest.mark.parametrize("name,sampler", sorted(DETERMINISTIC.items()))
def test_constant_field_is_exact(name, sampler):
    """Both deterministic solvers integrate a constant field exactly, at
    any step count."""
    c = 1.5
    torch.manual_seed(0)
    out = sampler(ConstantField(c), PATH, TARGET, SHAPE, torch.device("cpu"),
                  steps=3, progress=False)
    assert torch.allclose(out, _start_point(SHAPE) + c, atol=1e-5)


def test_euler_maruyama_reduces_to_euler_at_zero_noise():
    """sigma is the only thing separating the SDE from the probability-flow
    ODE, so sigma=0 must reproduce euler bit for bit. This is the check
    that the score-correction term is wired up with the right coefficient:
    a stray factor would survive every other test here."""
    from samplers import euler_maruyama

    torch.manual_seed(0)
    ode = euler(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                steps=10, progress=False)
    torch.manual_seed(0)
    sde = euler_maruyama(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                         steps=10, progress=False, sigma=0.0)
    assert torch.allclose(ode, sde, atol=1e-6)


def test_euler_maruyama_is_stochastic_but_finite():
    """sigma > 0 must actually inject noise -- and stay finite doing it,
    despite the score dividing by beta(t) -> 0 near the data endpoint."""
    from samplers import euler_maruyama

    torch.manual_seed(0)
    a = euler_maruyama(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                       steps=10, progress=False, sigma=0.3)
    torch.manual_seed(0)
    b = euler_maruyama(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                       steps=10, progress=False, sigma=0.0)
    assert torch.isfinite(a).all()
    assert not torch.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("steps", [4, 8])
def test_heun_is_second_order_where_euler_is_first(steps):
    """On v(x,t)=t the exact displacement is 1/2.

    Heun's trapezoid steps are exact for a field linear in t, and only its
    last step is plain Euler (see `heun`), so it undershoots by exactly
    dt^2/2 -- second order. Euler undershoots by dt/2 -- first order.
    Doubling the steps cuts Heun's error 4x and Euler's only 2x. This is
    the whole reason the sampler is a separate axis: swapping it changes
    the answer on a fixed model.
    """
    dt = 1 / steps
    start = _start_point(SHAPE)

    torch.manual_seed(0)
    e = euler(TimeRampField(), PATH, TARGET, SHAPE, torch.device("cpu"),
              steps=steps, progress=False)
    torch.manual_seed(0)
    h = heun(TimeRampField(), PATH, TARGET, SHAPE, torch.device("cpu"),
             steps=steps, progress=False)

    assert torch.allclose(h, start + 0.5 - dt**2 / 2, atol=1e-5)
    assert torch.allclose(e, start + 0.5 - dt / 2, atol=1e-5)


def test_heun_never_evaluates_the_network_at_the_data_endpoint():
    """A correction on the step onto t=1 would query the network where the
    learned field is least reliable, and averaging that in left visible
    noise in real samples. EDM skips it; so must we."""
    seen: list[float] = []

    class Recorder(nn.Module):
        def forward(self, x, t):
            seen.append(t[0].item())
            return torch.zeros_like(x)

    heun(Recorder(), PATH, TARGET, SHAPE, torch.device("cpu"), steps=5, progress=False)
    assert TARGET.t_range()[1] == 1.0, "test needs a grid that ends on the data endpoint"
    assert max(seen) < 1.0


class CallCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return torch.zeros_like(x)


@pytest.mark.parametrize("name", sorted(SAMPLERS))
@pytest.mark.parametrize("steps", [1, 2, 7])
def test_network_calls_match_what_the_sampler_really_does(name, steps):
    """Every equal-budget comparison is built on this count, so it is
    checked against the sampler itself rather than trusted."""
    counter = CallCounter()
    SAMPLERS[name](counter, PATH, TARGET, SHAPE, torch.device("cpu"), steps=steps, progress=False)
    assert counter.calls == network_calls(name, steps)


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_steps_for_budget_fits_as_many_steps_as_the_budget_allows(name):
    """Never over budget, and one more step would be: a solver given a
    budget uses all of it that its step size can."""
    for budget in range(1, 30):
        steps = steps_for_budget(name, budget)
        assert network_calls(name, steps) <= budget
        assert network_calls(name, steps + 1) > budget


def test_trajectory_shape():
    steps = 6
    torch.manual_seed(0)
    traj = euler(ConstantField(), PATH, TARGET, SHAPE, torch.device("cpu"),
                 steps=steps, progress=False, return_trajectory=True)
    assert traj.shape == (steps + 1, *SHAPE)
    assert torch.allclose(traj[0], _start_point(SHAPE), atol=1e-6)
