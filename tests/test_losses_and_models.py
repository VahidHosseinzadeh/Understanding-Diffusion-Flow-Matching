"""Tests for the loss and the two models."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dfm.losses import T_SAMPLERS, interpolant_loss, logit_normal_t, uniform_t
from dfm.mlp import MLP
from dfm.paths import LinearPath
from dfm.targets import VelocityTarget
from dfm.unet import UNet

PATH, TARGET = LinearPath(), VelocityTarget()


class ZeroModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


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
    doubled = interpolant_loss(ZeroModel(), x, PATH, TARGET, weighting=lambda t: 2.0 * torch.ones_like(t))
    assert torch.allclose(doubled, 2.0 * plain, atol=1e-6)


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
