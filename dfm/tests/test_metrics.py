"""Tests for the evaluation plumbing.

Nothing here runs Inception: that needs a 91 MB download and minutes of
compute, and would test torch-fidelity rather than this code. What is
tested is what torch-fidelity cannot check for us -- the pixel
conversion, the batching, and what gets cached -- because a mistake in
any of them still yields a perfectly plausible FID, for the wrong images.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset import _TRANSFORM
from metrics import UInt8Images, generate_images, image_metrics, to_uint8_rgb
from paths import LinearPath
from samplers import euler
from targets import VelocityTarget


def test_uint8_conversion_exactly_inverts_the_training_normalisation():
    """All 256 grey levels must survive stored uint8 -> the [-1, 1] tensor
    training sees -> uint8, bit for bit. Samples and real images then reach
    Inception quantised identically; one level off, and FID would partly
    measure the conversion instead of the model."""
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = to_uint8_rgb(_TRANSFORM(pixels).unsqueeze(0))  # the dataset's own transform

    assert rgb.dtype == torch.uint8 and rgb.shape == (1, 3, 16, 16)
    for channel in rgb[0]:  # grey repeated into all three
        assert torch.equal(channel, torch.from_numpy(pixels))


def test_out_of_range_samples_clamp_instead_of_wrapping():
    """Samples overshoot [-1, 1]. Cast unclamped, -1.2 wraps around to a
    near-white pixel and 1.2 to a near-black one."""
    x = torch.tensor([-3.0, -1.2, 1.2, 3.0]).reshape(1, 1, 2, 2)
    assert to_uint8_rgb(x)[0, 0].flatten().tolist() == [0, 0, 255, 255]


class Contract(nn.Module):
    """v = -x: the result depends on the starting noise, so seeds show."""

    def forward(self, x, t):
        return -x


def _generate(seed: int, n: int = 10) -> torch.Tensor:
    return generate_images(Contract(), LinearPath(), VelocityTarget(), euler, n=n,
                           sample_shape=(1, 4, 4), device=torch.device("cpu"), steps=3,
                           batch_size=4, seed=seed, progress=False)


def test_generate_images_returns_exactly_n_samples():
    """n need not divide by the batch size: the last, partial batch must be
    drawn, with nothing padded or dropped."""
    images = _generate(seed=0, n=10)  # batches of 4, 4, 2
    assert images.shape == (10, 3, 4, 4) and images.dtype == torch.uint8


def test_generate_images_is_reproducible_from_its_seed():
    """Same seed, same images -- whatever ran before. Otherwise two settings
    scored in one session do not start from the same noise, and part of
    their FID gap is just a different draw."""
    a = _generate(seed=0)
    torch.randn(100)  # disturb the global RNG
    assert torch.equal(a, _generate(seed=0))
    assert not torch.equal(a, _generate(seed=1))


def test_uint8_images_yields_bare_tensors():
    """torch-fidelity requires each item to *be* a uint8 3xHxW tensor."""
    ds = UInt8Images(torch.zeros(5, 3, 8, 8, dtype=torch.uint8))
    assert len(ds) == 5
    assert torch.is_tensor(ds[0]) and ds[0].shape == (3, 8, 8)


@pytest.fixture
def fidelity_call(monkeypatch) -> dict:
    """Stand in for torch-fidelity; returns the kwargs it was called with."""
    seen: dict = {}

    def fake_calculate_metrics(**kwargs):
        seen.update(kwargs)
        return {"frechet_inception_distance": 1.5}

    monkeypatch.setitem(sys.modules, "torch_fidelity",
                        types.SimpleNamespace(calculate_metrics=fake_calculate_metrics))
    return seen


SAMPLES = torch.zeros(50, 3, 8, 8, dtype=torch.uint8)
REFERENCE = torch.zeros(80, 3, 8, 8, dtype=torch.uint8)


def test_reference_is_cached_but_generated_samples_never_are(fidelity_call):
    """The caching mistake that matters: give the generated samples a cache
    name, and every later checkpoint is silently scored with the first
    one's features -- identical FIDs for different models."""
    out = image_metrics(SAMPLES, REFERENCE, "toy-ref", metrics=("fid", "kid"))

    assert out == {"frechet_inception_distance": 1.5}
    assert fidelity_call["input2_cache_name"] == "toy-ref"
    assert fidelity_call.get("input1_cache_name") is None
    assert fidelity_call["fid"] and fidelity_call["kid"]
    assert not fidelity_call["isc"] and not fidelity_call["prc"]
    assert fidelity_call["kid_subset_size"] <= 50  # KID's subsets cannot outnumber the samples
    with pytest.raises(ValueError, match="unknown metrics"):
        image_metrics(SAMPLES, REFERENCE, "toy-ref", metrics=("fdi",))


def test_runs_without_loader_workers(fidelity_call):
    """save_cpu_ram=True is what keeps precision/recall from materialising
    a 60k x 60k distance matrix, and what stops loader workers crashing on
    macOS when the caller has no __main__ guard. Neither shows in a test on
    small inputs, so pin the flag itself."""
    image_metrics(SAMPLES, REFERENCE, "toy-ref", metrics=("prc",))
    assert fidelity_call["save_cpu_ram"] is True


def test_missing_torch_fidelity_fails_with_an_install_hint(monkeypatch):
    """Only evaluation needs torch-fidelity, so its absence must surface as
    an install hint at the point of use -- never at import time."""
    monkeypatch.setitem(sys.modules, "torch_fidelity", None)  # makes `import` raise
    x = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    with pytest.raises(ImportError, match="pip install torch-fidelity"):
        image_metrics(x, x, "ref")
