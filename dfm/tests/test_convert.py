"""convert.py must put every foreign tensor where our model reads it.

The failures are silent ones: a transposed patch kernel or swapped
position axes loads without complaint and samples plausible-looking
noise. So the layouts are checked numerically against the operations
the reference code runs (a stride-p convolution, MAE's position codes),
and names are checked against the full list model_jit.py produces.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint, sample_shape
from convert import RENAMES, convert, convert_state, rename, to_our_layout
from jit import JIT_PRESETS, PatchEmbed, JiT, patchify, sincos_positions

# Li & He's architecture at toy size: their time scale and norm gains,
# a bottleneck, and class tokens joining at block 1 of 2.
TINY = dict(img_size=8, in_channels=3, patch_size=4, width=16, depth=2, heads=2,
            mlp_ratio=4.0, bottleneck_dim=6, in_context_tokens=3, in_context_start=1,
            time_freqs=8, time_scale=1.0, norm_gain=True)


def mae_positions(dim: int, grid: int) -> np.ndarray:
    """get_2d_sincos_pos_embed from MAE, which DiT and JiT copy verbatim."""
    def axis(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    g = np.arange(grid, dtype=np.float32)
    mesh = np.stack(np.meshgrid(g, g), axis=0)  # "here w goes first"
    return np.concatenate([axis(dim // 2, mesh[0]), axis(dim // 2, mesh[1])], axis=1)


def reference_state(config: dict, num_classes: int = 5) -> dict[str, torch.Tensor]:
    """Random tensors under the names and shapes model_jit.py gives them."""
    w, p, c, b = config["width"], config["patch_size"], config["in_channels"], config["bottleneck_dim"]
    hidden = int(int(config["mlp_ratio"] * w) * 2 / 3)
    head = w // config["heads"]
    shapes = {
        "t_embedder.mlp.0.weight": (w, config["time_freqs"]), "t_embedder.mlp.0.bias": (w,),
        "t_embedder.mlp.2.weight": (w, w), "t_embedder.mlp.2.bias": (w,),
        "y_embedder.embedding_table.weight": (num_classes + 1, w),
        "x_embedder.proj1.weight": (b, c, p, p),
        "x_embedder.proj2.weight": (w, b, 1, 1), "x_embedder.proj2.bias": (w,),
        "in_context_posemb": (1, config["in_context_tokens"], w),
        "final_layer.norm_final.weight": (w,),
        "final_layer.linear.weight": (p * p * c, w), "final_layer.linear.bias": (p * p * c,),
        "final_layer.adaLN_modulation.1.weight": (2 * w, w),
        "final_layer.adaLN_modulation.1.bias": (2 * w,),
    }
    block = {
        "norm1.weight": (w,), "norm2.weight": (w,),
        "attn.qkv.weight": (3 * w, w), "attn.qkv.bias": (3 * w,),
        "attn.q_norm.weight": (head,), "attn.k_norm.weight": (head,),
        "attn.proj.weight": (w, w), "attn.proj.bias": (w,),
        "mlp.w12.weight": (2 * hidden, w), "mlp.w12.bias": (2 * hidden,),
        "mlp.w3.weight": (w, hidden), "mlp.w3.bias": (w,),
        "adaLN_modulation.1.weight": (6 * w, w), "adaLN_modulation.1.bias": (6 * w,),
    }
    for i in range(config["depth"]):
        shapes.update({f"blocks.{i}.{k}": s for k, s in block.items()})
    gen = torch.Generator().manual_seed(0)
    state = {k: torch.randn(s, generator=gen) for k, s in shapes.items()}
    grid = config["img_size"] // p
    state["pos_embed"] = torch.from_numpy(mae_positions(w, grid)).float()[None]
    return state


def as_diffusers(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same weights under the Hugging Face mirror's names."""
    names = [(r"^final_layer\.norm_final\.", "norm_final."),
             (r"^final_layer\.linear\.", "linear_final."),
             (r"^final_layer\.adaLN_modulation\.1\.", "adaLN_modulation_final."),
             (r"\.adaLN_modulation\.1\.", ".adaLN_modulation.")]
    return rename(state, names)


def test_patch_convolution_becomes_our_linear():
    """Both of the bottleneck's convolutions, run as the reference runs
    them, against our PatchEmbed on patchify's vectors."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    proj1, proj2, bias = torch.randn(6, 3, 4, 4), torch.randn(16, 6, 1, 1), torch.randn(16)
    want = F.conv2d(F.conv2d(x, proj1, stride=4), proj2, bias).flatten(2).transpose(1, 2)

    state = to_our_layout({"patch_embed.proj.0.weight": proj1,
                           "patch_embed.proj.1.weight": proj2, "patch_embed.proj.1.bias": bias})
    embed = PatchEmbed(48, 16, bottleneck=6)
    embed.load_state_dict({k.removeprefix("patch_embed."): v for k, v in state.items()})
    assert torch.allclose(embed(patchify(x, 4)), want, atol=1e-5)


def test_position_codes_are_maes():
    """Same codes, same channel order: pretrained weights see what they
    were trained with. Rows and columns differ on a non-square grid of
    values, so a swap would show."""
    ours = sincos_positions(4, 4, 32)
    assert torch.allclose(ours, torch.from_numpy(mae_positions(32, 4)).float(), atol=1e-6)


@pytest.mark.parametrize("fmt", ["jit-official", "jit-diffusers"])
def test_every_reference_tensor_lands_in_place(fmt):
    ref = reference_state(TINY)
    source = {f"net.{k}": v for k, v in ref.items()} if fmt == "jit-official" else as_diffusers(ref)
    ours, model = convert_state(source, RENAMES[fmt], TINY)

    assert set(ours) == set(JiT(**TINY, num_classes=5).state_dict())
    assert model.config() == JiT(**TINY, num_classes=5).config()
    for theirs_name, our_name in [("blocks.1.adaLN_modulation.1.weight", "blocks.1.ada.weight"),
                                  ("blocks.0.attn.q_norm.weight", "blocks.0.attn.q_norm.weight"),
                                  ("final_layer.linear.bias", "final.out.bias"),
                                  ("t_embedder.mlp.2.weight", "time_emb.net.2.weight"),
                                  ("y_embedder.embedding_table.weight", "label_emb.table.weight")]:
        assert torch.equal(ours[our_name], ref[theirs_name])
    assert torch.equal(ours["context_pos"], ref["in_context_posemb"][0])


def test_an_unmapped_name_is_reported():
    source = reference_state(TINY)
    source["blocks.0.attn.mystery.weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="mystery"):
        convert_state(source, RENAMES["jit-diffusers"], TINY)


def test_swapped_position_axes_are_refused():
    source = reference_state(TINY)
    width = TINY["width"]
    pos = source["pos_embed"]
    source["pos_embed"] = torch.cat([pos[..., width // 2:], pos[..., :width // 2]], dim=-1)
    with pytest.raises(ValueError, match="position codes"):
        convert_state(source, RENAMES["jit-diffusers"], TINY)


def test_converted_checkpoint_loads_and_runs(tmp_path):
    """The whole route: an official-format file -> convert -> load_checkpoint
    -> a forward pass, with the raw weights and the chosen EMA kept apart."""
    raw = {f"net.{k}": v for k, v in reference_state(TINY).items()}
    ema = {k: v + 1 for k, v in raw.items() if not k.endswith("pos_embed")}
    ema["net.pos_embed"] = raw["net.pos_embed"]
    src = tmp_path / "checkpoint-last.pth"
    torch.save({"model": raw, "model_ema1": ema, "model_ema2": raw, "epoch": 599,
                "args": argparse.Namespace(model="JiT-tiny")}, src)

    out = tmp_path / "converted.pt"
    torch.save(convert("jit-official", str(src), TINY), out)
    loaded = load_checkpoint(str(out), torch.device("cpu"), use_ema=True)
    assert loaded.epoch == 599 and loaded.meta["num_classes"] == 5
    assert sample_shape(loaded.meta) == (3, 8, 8)
    assert torch.equal(loaded.model.blocks[0].ada.weight, ema["net.blocks.0.adaLN_modulation.1.weight"])
    raw_model = load_checkpoint(str(out), torch.device("cpu"), use_ema=False).model
    assert torch.equal(raw_model.blocks[0].ada.weight, raw["net.blocks.0.adaLN_modulation.1.weight"])

    x, t, y = torch.randn(2, 3, 8, 8), torch.rand(2), torch.tensor([0, 4])
    assert loaded.model(x, t, y).isfinite().all()


def test_b16_preset_matches_the_released_parameter_count():
    """525,298,808 bytes of float32 safetensors on the hub, less a 17,520-byte
    header, is 131,320,320 numbers; 196,608 of them are the fixed position
    codes, which we rebuild instead of storing."""
    with torch.device("meta"):
        model = JiT(**JIT_PRESETS["JiT-B/16"], num_classes=1000)
    assert sum(p.numel() for p in model.parameters()) == 131_320_320 - 256 * 768
