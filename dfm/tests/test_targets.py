"""Tests for the target axis.

The load-bearing property is that every target is a *reparameterisation*
of the same underlying object: give each one a perfect prediction and
they must all convert back to the identical velocity. A sign error or a
misplaced alpha/beta in one `to_velocity` would still train and still
produce plausible-looking samples, so only this identity catches it.
"""
from __future__ import annotations

import pytest
import torch

from paths import LinearPath
from targets import TARGETS, DataTarget, NoiseTarget, VelocityTarget

ALL_TARGETS = [VelocityTarget(), DataTarget(), NoiseTarget()]


@pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda tg: repr(tg))
@pytest.mark.parametrize("shape", [(16, 2), (4, 1, 28, 28)], ids=["2d", "image"])
def test_perfect_prediction_recovers_the_true_velocity(target, shape):
    """regression_target -> to_velocity must be the identity on velocity.

    This is the whole contract of the axis. It also exercises
    broadcasting: t is (B,) while x_t may be (B, C, H, W).
    """
    path = LinearPath()
    x_data, x_noise = torch.randn(shape), torch.randn(shape)
    t = torch.rand(shape[0]) * 0.8 + 0.1  # keep off the singular endpoints

    x_t = path.interpolate(x_data, x_noise, t)
    perfect_pred = target.regression_target(path, x_data, x_noise, t)
    recovered = target.to_velocity(path, x_t, t, perfect_pred)

    assert recovered.shape == x_data.shape
    assert torch.allclose(recovered, path.velocity(x_data, x_noise, t), atol=1e-4)


def test_all_targets_agree_with_each_other():
    """Stronger phrasing of the same idea: three parameterisations, one
    velocity field. Compares them pairwise rather than against the path."""
    path = LinearPath()
    x_data, x_noise = torch.randn(32, 2), torch.randn(32, 2)
    t = torch.rand(32) * 0.8 + 0.1
    x_t = path.interpolate(x_data, x_noise, t)

    velocities = [
        tg.to_velocity(path, x_t, t, tg.regression_target(path, x_data, x_noise, t))
        for tg in ALL_TARGETS
    ]
    for other in velocities[1:]:
        assert torch.allclose(velocities[0], other, atol=1e-4)


@pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda tg: repr(tg))
@pytest.mark.parametrize("t_val", [0.0, 1.0], ids=["t=0", "t=1"])
def test_endpoints_stay_finite(target, t_val):
    """alpha(0)=0 and beta(1)=0 make two of these inversions 0/0.

    The magnitude floor must keep that finite -- samplers evaluate at
    t=0 (euler) and at t=1 (heun's look-ahead on the final step), so a
    NaN here would poison an entire sampling run.
    """
    path = LinearPath()
    x_data, x_noise = torch.randn(8, 2), torch.randn(8, 2)
    t = torch.full((8,), t_val)

    x_t = path.interpolate(x_data, x_noise, t)
    pred = target.regression_target(path, x_data, x_noise, t)
    v = target.to_velocity(path, x_t, t, pred)

    assert torch.isfinite(v).all(), f"{target} produced non-finite velocity at t={t_val}"


def test_registry_is_wired_up():
    """`--target` on the CLI reads its choices straight from this dict."""
    assert set(TARGETS) == {"velocity", "x_data", "noise"}
    for name, cls in TARGETS.items():
        assert isinstance(cls(), (VelocityTarget, DataTarget, NoiseTarget))


@pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda tg: repr(tg))
def test_t_range_is_a_subinterval_of_unit_interval(target):
    lo, hi = target.t_range()
    assert 0.0 <= lo < hi <= 1.0


def test_velocity_target_uses_the_whole_interval():
    """Velocity inverts nothing, so it has no singular endpoint to avoid.
    Pinned because widening/narrowing it would silently change every
    existing rectified-flow result."""
    assert VelocityTarget().t_range() == (0.0, 1.0)


@pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda tg: repr(tg))
@pytest.mark.parametrize("sampler_name", ["euler", "heun"])
def test_samplers_never_evaluate_outside_t_range(target, sampler_name):
    """The whole point of `t_range`: a sampler must not step onto the
    endpoint where a target's inversion blows up. Recording the times
    the model actually sees is the only way to check this for real --
    asserting on the output would pass even if a NaN got clamped away.
    """
    import torch.nn as nn
    from samplers import SAMPLERS

    seen: list[float] = []

    class Recorder(nn.Module):
        def forward(self, x, t):
            seen.append(t[0].item())
            return torch.zeros_like(x)

    lo, hi = target.t_range()
    SAMPLERS[sampler_name](Recorder(), LinearPath(), target, (4, 2),
                           torch.device("cpu"), steps=10, progress=False)

    assert seen, "sampler never called the model"
    assert min(seen) >= lo - 1e-6, f"evaluated at t={min(seen)}, below {lo}"
    assert max(seen) <= hi + 1e-6, f"evaluated at t={max(seen)}, above {hi}"
