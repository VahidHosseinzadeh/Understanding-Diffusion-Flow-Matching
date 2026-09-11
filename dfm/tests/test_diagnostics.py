"""Tests for the diagnostics.

Each metric is checked against a case whose answer is known in closed
form, because a diagnostic that is silently wrong is worse than no
diagnostic -- it produces a confident plot you will trust.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from diagnostics import (
    LossTimeProfile,
    sampler_budget_matrix,
    straightness,
    velocity_norm_profile,
)
from paths import LinearPath
from samplers import NFE_PER_STEP
from targets import NoiseTarget, VelocityTarget


# --------------------------------------------------------------------
# straightness
# --------------------------------------------------------------------

def test_straight_line_has_straightness_exactly_one():
    """Equal steps along one direction: distance covered == path length."""
    steps = torch.linspace(0, 1, 11).reshape(11, 1, 1).expand(11, 4, 3)
    assert torch.allclose(straightness(steps.contiguous()), torch.ones(4), atol=1e-6)


def test_zigzag_straightness_matches_hand_computation():
    """0 -> 1 -> 0 -> 1 -> 2 travels 4 units to get 2 units away."""
    traj = torch.zeros(5, 1, 2)
    traj[:, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 1.0, 2.0])
    assert straightness(traj).item() == pytest.approx(0.5, abs=1e-6)


def test_straightness_handles_image_shaped_trajectories():
    traj = torch.randn(9, 6, 1, 8, 8)
    s = straightness(traj)
    assert s.shape == (6,)
    assert ((s > 0) & (s <= 1.0 + 1e-6)).all(), "straightness must lie in (0, 1]"


def test_straightness_rejects_a_non_trajectory():
    with pytest.raises(ValueError):
        straightness(torch.randn(1, 4, 2))  # only one point: no path


# --------------------------------------------------------------------
# LossTimeProfile
# --------------------------------------------------------------------

def test_profile_bins_by_t_and_averages_within_bin():
    p = LossTimeProfile(n_bins=4)
    p.update(torch.tensor([0.1, 0.2, 0.9]), torch.tensor([1.0, 3.0, 10.0]))
    centres, means, counts = p.summary()

    assert torch.allclose(centres, torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float64))
    assert means[0].item() == pytest.approx(2.0)   # (1 + 3) / 2
    assert means[3].item() == pytest.approx(10.0)
    assert counts.tolist() == [2.0, 0.0, 0.0, 1.0]


def test_empty_bins_are_nan_not_zero():
    """A bin nothing landed in is missing data. Plotting it as 0 would
    invent a dip in the loss profile that never happened."""
    p = LossTimeProfile(n_bins=4)
    p.update(torch.tensor([0.1]), torch.tensor([5.0]))
    _, means, _ = p.summary()
    assert torch.isnan(means[1:]).all()
    assert not torch.isnan(means[0])


def test_t_equal_to_one_lands_in_the_last_bin():
    """t == 1.0 would index out of range without the clamp."""
    p = LossTimeProfile(n_bins=4)
    p.update(torch.tensor([1.0]), torch.tensor([7.0]))
    _, means, counts = p.summary()
    assert counts[-1].item() == 1.0
    assert means[-1].item() == pytest.approx(7.0)


def test_reset_clears_history():
    p = LossTimeProfile(n_bins=4)
    p.update(torch.tensor([0.5]), torch.tensor([1.0]))
    assert not p.is_empty()
    p.reset()
    assert p.is_empty()


# --------------------------------------------------------------------
# velocity_norm_profile
# --------------------------------------------------------------------

class ConstantVelocity(nn.Module):
    def forward(self, x, t):
        return torch.ones_like(x)


def test_velocity_norm_profile_shapes_and_flatness():
    times, mean_n, max_n = velocity_norm_profile(
        ConstantVelocity(), LinearPath(), VelocityTarget(),
        torch.randn(16, 3), n_times=7,
    )
    assert times.shape == mean_n.shape == max_n.shape == (7,)
    # ||(1,1,1)|| = sqrt(3) at every t, so the profile must be flat
    assert torch.allclose(mean_n, torch.full((7,), 3.0 ** 0.5), atol=1e-5)
    assert (max_n >= mean_n).all()


def test_velocity_norm_profile_detects_a_boundary_explosion():
    """The diagnostic's whole purpose. NoiseTarget's to_velocity divides by
    alpha(t), which vanishes at t=0, so the profile must blow up there and
    stay tame elsewhere."""
    _, mean_n, _ = velocity_norm_profile(
        ConstantVelocity(), LinearPath(), NoiseTarget(),
        torch.randn(16, 3), n_times=21,
    )
    assert mean_n[0] > 100 * mean_n[-1], "explosion at t=0 not detected"


# --------------------------------------------------------------------
# sampler budget matrix
# --------------------------------------------------------------------

def test_budget_matrix_equalises_network_calls_not_steps():
    """Heun evaluates the field twice per step, so at a 20-call budget it
    must take 10 steps while euler takes 20. Comparing at equal steps
    quietly gives heun double the compute -- the usual way this
    comparison is reported wrongly."""
    calls: dict[str, int] = {}

    class Counter(nn.Module):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def forward(self, x, t):
            calls[self.key] = calls.get(self.key, 0) + 1
            return torch.zeros_like(x)

    for name in ("euler", "heun", "euler_maruyama"):
        sampler_budget_matrix(
            Counter(name), LinearPath(), VelocityTarget(), (4, 2),
            torch.device("cpu"), budgets=(20,), sampler_names=(name,),
        )

    for name, n in calls.items():
        assert n == 20, f"{name} used {n} network calls, expected 20"
        assert NFE_PER_STEP[name] in (1, 2)


def test_budget_matrix_covers_every_cell():
    grids = sampler_budget_matrix(
        ConstantVelocity(), LinearPath(), VelocityTarget(), (4, 2),
        torch.device("cpu"), budgets=(4, 8),
        sampler_names=("euler", "heun"),
    )
    assert set(grids) == {("euler", 4), ("euler", 8), ("heun", 4), ("heun", 8)}
    for v in grids.values():
        assert v.shape == (4, 2) and torch.isfinite(v).all()
