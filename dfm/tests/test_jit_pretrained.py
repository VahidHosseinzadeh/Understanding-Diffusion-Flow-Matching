"""Li & He's code is copied, not tested; what is tested is that the two
copies differ only where they should, and the parts that are ours:
label translation, telling their models apart, loading their files.

Bit-identity of the copies with the untouched original, on the real
JiT-B/16 checkpoint, was checked when they were made (it needs their
repo and the 1.5 GB file, so it is not a test here).
"""
from __future__ import annotations

import difflib
from functools import partial
from pathlib import Path

import pytest
import torch

import jit_mac
import jit_official
import jit_pretrained
from checkpoint import load_checkpoint, sample_shape
from embeddings import NULL_LABEL
from jit_pretrained import SIGNATURE, OfficialJiT, identify
from paths import LinearPath
from targets import DataTarget

DFM = Path(__file__).resolve().parent.parent

# Their architecture at toy size: 8x8 images, 4x4 patches, two blocks,
# two class tokens joining at block 1.
TINY = "JiT-tiny/4"
TINY_KWARGS = dict(patch_size=4, hidden_size=32, depth=2, num_heads=2,
                   bottleneck_dim=8, in_context_len=2, in_context_start=1)


@pytest.fixture
def tiny(monkeypatch):
    """Make TINY one of 'their' models, and the only one, in both copies,
    and return it built from the .cuda()-free copy on the CPU."""
    for code in (jit_mac, jit_official):
        monkeypatch.setattr(code, "JiT_models", {TINY: partial(code.JiT, **TINY_KWARGS)})
    monkeypatch.setattr(jit_pretrained, "SIGNATURE", {TINY: (2, 32, 4)})
    monkeypatch.setattr(jit_pretrained, "IMG_SIZE", {TINY: 8})
    monkeypatch.setattr(jit_pretrained, "NOISE_SCALE", {TINY: 1.0})
    monkeypatch.setattr(torch._dynamo.config, "disable", True)  # no compiling in tests
    torch.manual_seed(0)
    model = OfficialJiT(TINY, num_classes=5, code=jit_mac)
    with torch.no_grad():  # their init zeroes the output layer; random weights see everything
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    return model.eval()


def _code(name: str) -> list[str]:
    text = (DFM / f"{name}.py").read_text()
    return text[text.index("# util/model_util.py"):].splitlines()  # past the docstring


def test_the_mac_copy_differs_only_in_cuda_lines():
    """Every line jit_mac.py changes was a CUDA call in jit_official.py,
    and every replacement says so."""
    diff = list(difflib.ndiff(_code("jit_official"), _code("jit_mac")))
    removed = [line[2:] for line in diff if line.startswith("- ")]
    added = [line[2:] for line in diff if line.startswith("+ ")]
    assert len(removed) == len(added) == 6
    assert all(".cuda()" in line or "torch.cuda" in line for line in removed)
    assert all("# dfm: was" in line and "cuda" not in line.split("# dfm")[0] for line in added)


def test_rotate_half_without_einops_turns_each_pair():
    """einops' rearrange replaced by torch: every pair (a, b) -> (-b, a)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for code in (jit_official, jit_mac):
        assert torch.equal(code.rotate_half(x), torch.tensor([-2.0, 1.0, -4.0, 3.0]))


def test_rope_tables_without_einops_are_their_formula():
    """einops' repeat replaced by torch: pair i of the token at (row, col)
    turns by row * f_i in the first half of the channels and col * f_i in
    the second, both channels of a pair together, f_i = 10000^(-2i/dim)."""
    rope = jit_mac.VisionRotaryEmbeddingFast(dim=4, pt_seq_len=3)  # 2 frequencies per axis
    f = 10000.0 ** (-torch.arange(0, 4, 2).float() / 4)
    for row in range(3):
        for col in range(3):
            angle = torch.cat([row * f, col * f]).repeat_interleave(2)
            assert torch.allclose(rope.freqs_cos[3 * row + col], angle.cos())
            assert torch.allclose(rope.freqs_sin[3 * row + col], angle.sin())


def test_signatures_are_their_constructors(monkeypatch):
    """SIGNATURE copies depth, width and patch size out of model_jit.py's
    JiT_B_16 etc.; read them back from the constructors themselves."""
    monkeypatch.setattr(jit_official, "JiT", lambda **kwargs: kwargs)  # record, do not build
    for name, build in jit_official.JiT_models.items():
        kwargs = build()
        assert SIGNATURE[name] == (kwargs["depth"], kwargs["hidden_size"], kwargs["patch_size"])


def test_identify_reads_the_signature_off_the_weights():
    for name, (depth, width, patch) in SIGNATURE.items():
        state = {f"net.blocks.{i}.norm1.weight": torch.empty(width) for i in range(depth)}
        state["net.pos_embed"] = torch.empty(1, 256, width)
        state["net.x_embedder.proj1.weight"] = torch.empty(128, 3, patch, patch)
        assert identify(state) == name
    state["net.pos_embed"] = torch.empty(1, 256, 640)
    with pytest.raises(ValueError, match="width 640"):
        identify(state)


def test_no_label_is_their_null_row(tiny):
    x, t = torch.randn(3, 3, 8, 8), torch.tensor([0.1, 0.5, 0.9])
    with torch.no_grad():
        theirs = tiny.net(x, t, torch.full((3,), 5))  # their "no label" is index num_classes
        assert torch.allclose(tiny(x, t), theirs, atol=1e-6)
        assert torch.allclose(tiny(x, t, torch.full((3,), NULL_LABEL)), theirs, atol=1e-6)

        y = torch.tensor([0, 4, 2])
        assert torch.allclose(tiny(x, t, y), tiny.net(x, t, y), atol=1e-6)
        assert not torch.allclose(tiny(x, t, y), theirs, atol=1e-3)  # the label matters
        with pytest.raises(ValueError):
            tiny(x, t, torch.tensor([0, 5, 2]))


def _their_file(tmp_path, tiny):
    """Their checkpoint-last.pth layout: the Denoiser's state dicts, with
    the network under `net.`, and no meta."""
    raw = tiny.state_dict()
    ema1 = {k: v + 1 for k, v in raw.items()}
    file = tmp_path / "checkpoint-last.pth"
    torch.save({"model": raw, "model_ema1": ema1, "model_ema2": raw}, file)
    return file, raw, ema1


@pytest.mark.parametrize("use_ema", [True, False], ids=["ema1", "raw"])
def test_load_checkpoint_reads_their_file(tmp_path, tiny, use_ema):
    file, raw, ema1 = _their_file(tmp_path, tiny)
    loaded = load_checkpoint(str(file), torch.device("cpu"), use_ema=use_ema)
    want = ema1 if use_ema else raw
    assert all(torch.equal(loaded.model.state_dict()[k], want[k]) for k in want)
    assert type(loaded.model.net).__module__ == "jit_mac"  # no CUDA, so the .cuda()-free copy
    assert isinstance(loaded.path, LinearPath) and loaded.path.beta_min == 0.0
    assert isinstance(loaded.target, DataTarget)
    assert loaded.meta["jit_official"] == TINY and loaded.meta["num_classes"] == 5
    assert sample_shape(loaded.meta) == (3, 8, 8)
    assert not loaded.model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_on_a_gpu_their_own_code_runs(tmp_path, tiny):
    file, _, ema1 = _their_file(tmp_path, tiny)
    loaded = load_checkpoint(str(file), torch.device("cuda"))
    assert type(loaded.model.net).__module__ == "jit_official"
    x, t = torch.randn(2, 3, 8, 8), torch.tensor([0.2, 0.8])
    with torch.no_grad():
        on_gpu = loaded.model(x.cuda(), t.cuda(), torch.tensor([1, NULL_LABEL]).cuda()).cpu()
        tiny.load_state_dict(ema1)
        on_cpu = tiny(x, t, torch.tensor([1, NULL_LABEL]))
    assert torch.allclose(on_gpu, on_cpu, atol=1e-4)  # the two copies compute the same thing


def test_noise_scaled_models_are_refused(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(jit_pretrained, "NOISE_SCALE", {TINY: 2.0})
    file, _, _ = _their_file(tmp_path, tiny)
    with pytest.raises(SystemExit, match="noise"):
        load_checkpoint(str(file), torch.device("cpu"))
