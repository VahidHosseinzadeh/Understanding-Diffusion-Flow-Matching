"""load_checkpoint must rebuild exactly the model that was trained.

The failures it guards against are silent: raw weights loaded where EMA
was meant (or the reverse) still sample plausibly, and an evaluation of
the wrong weights still produces a plausible FID.
"""
from __future__ import annotations

import pytest
import torch

from checkpoint import load_checkpoint, sample_shape
from mlp import MLP
from paths import LinearPath
from targets import DataTarget

META = {"path": "linear", "beta_min": 0.01, "target": "x_data", "model": "mlp",
        "data": "moons", "hidden": 16, "depth": 2, "base_channels": 64}


def _save(tmp_path):
    torch.manual_seed(0)
    raw, ema = MLP(dim=2, hidden=16, depth=2), MLP(dim=2, hidden=16, depth=2)  # two inits
    file = tmp_path / "checkpoint.pt"
    torch.save({"model": raw.state_dict(), "ema": ema.state_dict(), "epoch": 7, "meta": META}, file)
    return file, raw.state_dict(), ema.state_dict()


@pytest.mark.parametrize("use_ema", [True, False], ids=["ema", "raw"])
def test_loads_the_weights_it_was_asked_for(tmp_path, use_ema):
    # Compared weight by weight, not by output: the MLP zero-initialises its
    # output layer, so two fresh models agree (on 0) however they differ.
    file, raw, ema = _save(tmp_path)
    model = load_checkpoint(str(file), torch.device("cpu"), use_ema=use_ema).model
    loaded = model.state_dict()
    wanted, other = (ema, raw) if use_ema else (raw, ema)

    assert all(torch.equal(loaded[k], wanted[k]) for k in wanted)
    assert not all(torch.equal(loaded[k], other[k]) for k in other)
    assert not model.training


def test_rebuilds_path_and_target_from_meta(tmp_path):
    file, _, _ = _save(tmp_path)
    loaded = load_checkpoint(str(file), torch.device("cpu"))
    assert isinstance(loaded.path, LinearPath) and loaded.path.beta_min == 0.01
    assert isinstance(loaded.target, DataTarget)
    assert loaded.epoch == 7 and loaded.meta == META


def test_sample_shape_follows_the_architecture():
    assert sample_shape({"model": "mlp"}) == (2,)
    assert sample_shape({"model": "unet"}) == (1, 28, 28)


def test_refuses_a_checkpoint_without_meta(tmp_path):
    """Guessing the path and target would decode with the wrong ones."""
    file = tmp_path / "old.pt"
    torch.save({"model": {}, "ema": {}}, file)
    with pytest.raises(SystemExit, match="meta"):
        load_checkpoint(str(file), torch.device("cpu"))
