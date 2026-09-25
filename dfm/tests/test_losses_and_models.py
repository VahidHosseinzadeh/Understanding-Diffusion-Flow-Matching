"""Tests for the loss and the two models."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from losses import T_SAMPLERS, LossSpaceWeighting, interpolant_loss, logit_normal_t, uniform_t
from mlp import MLP
from paths import LinearPath, expand_to
from targets import TARGETS, DataTarget, NoiseTarget, VelocityTarget
from unet import UNet

PATH, TARGET = LinearPath(), VelocityTarget()


class ZeroModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class Affine(nn.Module):
    """A fixed, deliberately wrong predictor: errors that vary with x and t."""

    def forward(self, x, t):
        return 0.5 * x + expand_to(t, x)


def test_loss_is_finite_and_backprops():
    model = MLP(hidden=32, depth=2)
    loss = interpolant_loss(model, torch.randn(16, 2), PATH, TARGET)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_model_loss_equals_expected_target_energy():
    """A model predicting 0 must incur exactly mean(||v_target||^2).

    For the linear path v_target = x_data - x_noise, so this pins down
    the loss end to end -- interpolation, target and reduction -- against
    a value computed independently of the library code.
    """
    torch.manual_seed(0)
    x_data = torch.randn(256, 2)

    torch.manual_seed(1)
    loss = interpolant_loss(ZeroModel(), x_data, PATH, TARGET)

    torch.manual_seed(1)
    x_noise = torch.randn_like(x_data)          # same draw order as the loss
    _ = uniform_t(x_data.shape[0], x_data.device)
    expected = (x_data - x_noise).pow(2).flatten(1).mean(dim=1).mean()

    assert torch.allclose(loss, expected, atol=1e-6)


def test_weighting_scales_the_loss():
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    torch.manual_seed(3)
    plain = interpolant_loss(ZeroModel(), x, PATH, TARGET)
    torch.manual_seed(3)
    doubled = interpolant_loss(ZeroModel(), x, PATH, TARGET,
                               weighting=lambda path, target, t: 2.0 * torch.ones_like(t))
    assert torch.allclose(doubled, 2.0 * plain, atol=1e-6)


# --------------------------------------------------------------------
# loss space: where the error is measured, independent of the target
# --------------------------------------------------------------------

def _interior_t(batch: int, device: torch.device) -> torch.Tensor:
    """U[0.1, 0.9]: off the singular endpoints, so no weight reaches a cap."""
    return torch.rand(batch, device=device) * 0.8 + 0.1


@pytest.mark.parametrize("target_name", sorted(TARGETS))
@pytest.mark.parametrize("space_name", sorted(TARGETS))
def test_loss_space_weighting_equals_converting_the_prediction(target_name, space_name):
    """The error-conversion table, checked cell by cell.

    Predicting `target` and reweighting into `space` must give the same
    loss as converting the prediction into `space` and taking a plain MSE
    there. The conversion goes to_velocity -> Path.solve -> the space's
    own regression_target, so it shares no code with the weights.
    """
    path, model = LinearPath(), Affine()
    target, space = TARGETS[target_name](), TARGETS[space_name]()
    torch.manual_seed(0)
    x_data = torch.randn(64, 2)

    torch.manual_seed(1)
    weighted = interpolant_loss(model, x_data, path, target, t_sampler=_interior_t,
                                weighting=LossSpaceWeighting(space, max_weight=float("inf")))

    torch.manual_seed(1)
    x_noise = torch.randn_like(x_data)          # same draw order as the loss
    t = _interior_t(x_data.shape[0], x_data.device)
    x_t = path.interpolate(x_data, x_noise, t)
    v_pred = target.to_velocity(path, x_t, t, model(x_t, t))
    pred_in_space = space.regression_target(path, *path.solve(x_t, v_pred, t), t)
    true_in_space = space.regression_target(path, x_data, x_noise, t)
    direct = (pred_in_space - true_in_space).pow(2).flatten(1).mean(dim=1).mean()

    assert torch.allclose(weighted, direct, rtol=1e-4)


def test_loss_space_weights_match_their_closed_forms_on_the_linear_path():
    """Three cells worked by hand, with alpha = t, beta = 1 - t, det = -1."""
    path = LinearPath()
    t = torch.linspace(0.1, 0.9, 9)

    def w(target, space):
        return LossSpaceWeighting(space, max_weight=float("inf"))(path, target, t)

    assert torch.allclose(w(DataTarget(), VelocityTarget()), 1 / (1 - t) ** 2)
    assert torch.allclose(w(NoiseTarget(), VelocityTarget()), 1 / t**2)
    assert torch.allclose(w(DataTarget(), NoiseTarget()), t**2 / (1 - t) ** 2)  # SNR(t)


def test_matches_jits_own_loss_code():
    """JiT (github.com/LTH14/JiT, denoiser.py) never reweights: it converts
    both sides to velocity and takes a plain MSE,

        v      = (x - z) / (1 - t).clamp_min(t_eps)
        v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)
        loss   = (v - v_pred) ** 2

    z cancels in v - v_pred, so that is the x_data loss times
    min(1/(1-t)^2, 1/t_eps^2) -- LossSpaceWeighting at its default cap,
    clipped samples included. Transcribed line for line so the
    equivalence is checked rather than argued.
    """
    path, model, t_eps = LinearPath(), Affine(), 5e-2  # JiT's default --t_eps
    torch.manual_seed(0)
    x = torch.randn(256, 2)

    torch.manual_seed(1)
    ours = interpolant_loss(model, x, path, DataTarget(),
                            weighting=LossSpaceWeighting(VelocityTarget()))

    torch.manual_seed(1)
    e = torch.randn_like(x)                     # same draw order as the loss
    t = uniform_t(x.shape[0], x.device).view(-1, 1)
    assert (t > 1 - t_eps).any(), "no sample landed in the clipped region"
    z = t * x + (1 - t) * e
    v = (x - z) / (1 - t).clamp_min(t_eps)
    v_pred = (model(z, t.flatten()) - z) / (1 - t).clamp_min(t_eps)
    jit = ((v - v_pred) ** 2).mean(dim=1).mean()

    assert torch.allclose(ours, jit, rtol=1e-4)


@pytest.mark.parametrize("target_name", sorted(TARGETS))
@pytest.mark.parametrize("space_name", sorted(TARGETS))
def test_loss_space_weight_is_finite_and_capped_at_the_endpoints(target_name, space_name):
    """alpha(0) = 0 and beta(1) = 0 make half of these ratios singular, and
    uniform t gets arbitrarily close to both. The weight must stay finite
    and within [0, max_weight] -- never NaN, which an unfloored inf/inf
    would give a target measured in its own space."""
    t = torch.tensor([0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0])
    weighting = LossSpaceWeighting(TARGETS[space_name](), max_weight=400.0)
    w = weighting(LinearPath(), TARGETS[target_name](), t)
    assert torch.isfinite(w).all()
    assert ((w >= 0) & (w <= 400.0)).all()


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_measuring_a_target_in_its_own_space_changes_nothing(name):
    t = torch.linspace(0.0, 1.0, 11)
    w = LossSpaceWeighting(TARGETS[name]())(LinearPath(), TARGETS[name](), t)
    assert torch.equal(w, torch.ones_like(t))


@pytest.mark.parametrize("name", sorted(T_SAMPLERS))
def test_t_samplers_stay_in_unit_interval(name):
    t = T_SAMPLERS[name](4096, torch.device("cpu"))
    assert t.shape == (4096,)
    assert (t >= 0).all() and (t <= 1).all()


def test_logit_normal_concentrates_away_from_endpoints():
    """The SD3 t-distribution should put less mass near 0 and 1 than uniform."""
    torch.manual_seed(0)
    ln = logit_normal_t(20000, torch.device("cpu"))
    un = uniform_t(20000, torch.device("cpu"))
    edge = lambda t: ((t < 0.1) | (t > 0.9)).float().mean()
    assert edge(ln) < edge(un)


@pytest.mark.parametrize(
    "model,shape",
    [
        (MLP(dim=2, hidden=32, depth=2), (8, 2)),
        (UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1), (4, 1, 28, 28)),
    ],
)
def test_models_honour_the_forward_contract(model, shape):
    """forward(x, t) -> same shape as x, with t a float tensor in [0, 1]."""
    x = torch.randn(shape)
    out = model(x, torch.rand(shape[0]))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
