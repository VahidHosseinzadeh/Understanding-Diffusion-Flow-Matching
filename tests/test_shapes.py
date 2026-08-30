"""Fast sanity checks (no dataset download, no real training): shapes,
numerical ranges and basic invariants for the UNet and both processes.
Run with: pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dfm.ddpm import DDPM, cosine_beta_schedule, linear_beta_schedule
from dfm.flow_matching import RectifiedFlow
from dfm.unet import UNet


def _tiny_model():
    return UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1)


def test_unet_forward_shape():
    model = _tiny_model()
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    out = model(x, t)
    assert out.shape == x.shape


def test_beta_schedules_are_valid():
    for betas in (linear_beta_schedule(100), cosine_beta_schedule(100)):
        assert betas.shape == (100,)
        assert (betas > 0).all() and (betas < 1).all()
        # betas should generally increase (or at least not blow up) across the schedule
        assert torch.isfinite(betas).all()


def test_ddpm_q_sample_shape_and_endpoints():
    ddpm = DDPM(timesteps=100, schedule="cosine")
    x0 = torch.randn(4, 1, 28, 28)
    # t=0 should barely perturb x0
    t0 = torch.zeros(4, dtype=torch.long)
    x_t0 = ddpm.q_sample(x0, t0, noise=torch.zeros_like(x0))
    assert torch.allclose(x_t0, x0 * ddpm.sqrt_alphas_bar[0], atol=1e-5)
    # shape sanity at a mid timestep
    t_mid = torch.full((4,), 50, dtype=torch.long)
    x_t_mid = ddpm.q_sample(x0, t_mid)
    assert x_t_mid.shape == x0.shape


def test_ddpm_training_loss_is_finite_and_backprop_works():
    model = _tiny_model()
    ddpm = DDPM(timesteps=100, schedule="cosine")
    x0 = torch.randn(4, 1, 28, 28)
    loss = ddpm.training_loss(model, x0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_ddpm_sample_shape():
    model = _tiny_model()
    ddpm = DDPM(timesteps=10, schedule="cosine")  # few steps, just checking shapes
    out = ddpm.sample(model, (2, 1, 28, 28), device=torch.device("cpu"), progress=False)
    assert out.shape == (2, 1, 28, 28)


def test_ddpm_ddim_sample_shape():
    model = _tiny_model()
    ddpm = DDPM(timesteps=50, schedule="cosine")
    out = ddpm.ddim_sample(model, (2, 1, 28, 28), device=torch.device("cpu"), steps=5, progress=False)
    assert out.shape == (2, 1, 28, 28)


def test_flow_matching_training_loss_and_sample():
    model = _tiny_model()
    rf = RectifiedFlow()
    x1 = torch.randn(4, 1, 28, 28)
    loss = rf.training_loss(model, x1)
    assert torch.isfinite(loss)
    loss.backward()

    out = rf.sample(model, (2, 1, 28, 28), device=torch.device("cpu"), steps=5, progress=False)
    assert out.shape == (2, 1, 28, 28)


def test_ema_tracks_model():
    from dfm.utils import EMA

    model = _tiny_model()
    ema = EMA(model, decay=0.0)  # decay=0 -> EMA should instantly match model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for p_model, p_ema in zip(model.parameters(), ema.module.parameters()):
        assert torch.allclose(p_model, p_ema)
