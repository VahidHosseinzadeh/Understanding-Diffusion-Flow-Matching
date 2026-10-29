#!/usr/bin/env python
"""Bring weights trained with someone else's code into this package.

A converter turns a foreign checkpoint into an ordinary checkpoint of
ours: weights for one of our models plus the `meta` block from which
`checkpoint.load_checkpoint` rebuilds model, path and target. After that
nothing downstream can tell where the weights came from -- sample.py and
every sampler run on them unchanged.

    python dfm/convert.py jit-official checkpoint-last.pth --preset JiT-B/16 --out jit_b16.pt
    python dfm/sample.py --checkpoint jit_b16.pt --sampler heun --steps 50 \\
        --cfg-scale 3 --guidance-interval 0.1 1 --label 207 360 387 974 --n 16

Three things have to line up, and each is checked rather than assumed:

  names     a format is a list of regex renames from its parameter names
            to ours. A name no rule maps, or one of ours that nothing
            fills, stops the conversion with both lists printed.
  layouts   a convolution over p x p patches is the same map as our
            Linear on `patchify`'s vectors, once its weight is reordered
            to (row, column, channel). Fixed position codes are not
            loaded but compared with ours, which catches a swapped axis.
  meaning   what weights cannot show: which way time runs, how t is
            scaled before it is embedded, the path, the noise level, what
            the network predicts. Li & He's JiT runs from t = 0 noise to
            t = 1 data on the straight path, as everything here does, and
            predicts x_data; its time scale and norm gains are options
            of our JiT, set in `JIT_PRESETS`.

Weights from another JiT implementation need one more list of renames
in RENAMES and a reader in READERS. Another architecture (DiT, SiT)
first needs its differences as options of a model of ours, as
`time_scale` and `norm_gain` are for JiT.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable

import torch

from jit import JIT_PRESETS, JiT, sincos_positions

StateDict = dict[str, torch.Tensor]
Renames = list[tuple[str, str]]  # (regex, replacement), applied to every name in order

# Names in Li & He's model_jit.py (github.com/LTH14/JiT) that differ from
# ours. The rest -- blocks.N.norm1, .attn.qkv, .attn.q_norm, .mlp.w12, ... --
# are the same in both.
_JIT_COMMON: Renames = [
    (r"^t_embedder\.mlp\.", "time_emb.net."),
    (r"^y_embedder\.embedding_table\.", "label_emb.table."),
    (r"^x_embedder\.proj1\.", "patch_embed.proj.0."),
    (r"^x_embedder\.proj2\.", "patch_embed.proj.1."),
    (r"^in_context_posemb$", "context_pos"),
    (r"^pos_embed$", "pos"),  # compared with ours, not loaded
]

RENAMES: dict[str, Renames] = {
    # The authors' release: training checkpoints of their Denoiser, which
    # holds the JiT as `net`.
    "jit-official": [(r"^net\.", "")] + _JIT_COMMON + [
        (r"^blocks\.(\d+)\.adaLN_modulation\.1\.", r"blocks.\1.ada."),
        (r"^final_layer\.norm_final\.", "final.norm."),
        (r"^final_layer\.linear\.", "final.out."),
        (r"^final_layer\.adaLN_modulation\.1\.", "final.ada."),
    ],
    # BiliSakura/JiT-diffusers on the Hugging Face hub: the same names, with
    # the adaLN Sequential flattened and the final layer's parts moved up.
    "jit-diffusers": _JIT_COMMON + [
        (r"^blocks\.(\d+)\.adaLN_modulation\.", r"blocks.\1.ada."),
        (r"^norm_final\.", "final.norm."),
        (r"^linear_final\.", "final.out."),
        (r"^adaLN_modulation_final\.", "final.ada."),
    ],
}


# --------------------------------------------------------------------
# readers: file -> (raw weights, EMA weights, epoch)
# --------------------------------------------------------------------

def read_official(file: str, ema: str) -> tuple[StateDict, StateDict, int | None]:
    """checkpoint-last.pth from the authors: raw weights under 'model' and
    two EMAs, 'model_ema1' (decay 0.9999, the one they sample with) and
    'model_ema2' (0.9996). `weights_only` keeps the pickle from running
    code; their saved argparse arguments are the one class it must allow."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(file, map_location="cpu", weights_only=True)
    return ckpt["model"], ckpt[f"model_{ema}"], ckpt.get("epoch")


def read_diffusers(file: str, ema: str) -> tuple[StateDict, StateDict, int | None]:
    """A single set of weights, which the mirror does not label. For
    JiT-B/16 it is the authors' model_ema1, bit for bit."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("reading .safetensors needs: pip install safetensors") from exc
    state = load_file(file)
    return state, state, None


READERS: dict[str, Callable[[str, str], tuple[StateDict, StateDict, int | None]]] = {
    "jit-official": read_official,
    "jit-diffusers": read_diffusers,
}


# --------------------------------------------------------------------
# the conversion
# --------------------------------------------------------------------

def rename(state: StateDict, renames: Renames) -> StateDict:
    out = {}
    for name, value in state.items():
        for pattern, replacement in renames:
            name = re.sub(pattern, replacement, name)
        out[name] = value
    return out


def to_our_layout(state: StateDict) -> StateDict:
    """Reshape renamed tensors into our modules' layouts.

    A convolution weight (out, C, kh, kw) applied with stride equal to its
    kernel is a Linear on each patch's pixels; ours reads them in
    `patchify`'s order (row, column, channel), hence the permute. The
    bottleneck's 1x1 convolution is the same rule with kh = kw = 1.
    """
    out = dict(state)
    for name, value in state.items():
        if name.startswith("patch_embed.") and value.ndim == 4:
            out[name] = value.permute(0, 2, 3, 1).reshape(value.shape[0], -1)
    if "context_pos" in out and out["context_pos"].ndim == 3:
        out["context_pos"] = out["context_pos"].squeeze(0)  # (1, L, D) -> (L, D)
    return out


def check_positions(theirs: torch.Tensor, ours: torch.Tensor) -> None:
    """Fixed position codes are rebuilt from the config, not loaded, so a
    source that orders them differently would load without complaint and
    sample nonsense. Compare instead.

    The tolerance is loose on purpose. Li & He's EMA averages every
    parameter, the frozen codes included, and averaging a constant some
    10^5 times in float32 drifts: their ema1 codes are off by 1.6e-4, against
    0 in their raw weights. A swapped axis or a wrong grid is off by ~1.
    """
    theirs = theirs.reshape(ours.shape).to(ours)
    if not torch.allclose(theirs, ours, atol=1e-3):
        raise ValueError(
            "the source's fixed position codes differ from ours (max difference "
            f"{(theirs - ours).abs().max():.3g}): a wrong preset, or rows and "
            "columns in the other order"
        )


def convert_state(state: StateDict, renames: Renames, config: dict[str, Any]) -> tuple[StateDict, JiT]:
    """One foreign state dict -> one of ours for `JiT(**config)`.

    Also returns the model it fits, built on the meta device: names,
    shapes and config only, no memory for weights. The number of classes
    is read off the label table, whose last row is "no label".
    """
    state = to_our_layout(rename(state, renames))
    num_classes = state["label_emb.table.weight"].shape[0] - 1
    with torch.device("meta"):
        model = JiT(**config, num_classes=num_classes)
    if "pos" in state:
        grid = config["img_size"] // config["patch_size"]
        check_positions(state.pop("pos"), sincos_positions(grid, grid, config["width"]))

    ours = model.state_dict()
    unmapped, unfilled = sorted(set(state) - set(ours)), sorted(set(ours) - set(state))
    if unmapped or unfilled:
        raise ValueError(
            "names do not line up; add renames for this format.\n"
            f"  source names no rule maps: {unmapped}\n"
            f"  our names nothing fills:   {unfilled}"
        )
    wrong = [f"{k}: {tuple(state[k].shape)} vs ours {tuple(v.shape)}"
             for k, v in ours.items() if state[k].shape != v.shape]
    if wrong:
        raise ValueError("shapes do not match the config:\n  " + "\n  ".join(wrong))
    return {k: state[k].to(v.dtype).contiguous() for k, v in ours.items()}, model


def convert(fmt: str, file: str, config: dict[str, Any], ema: str = "ema1",
            data: str = "imagenet256", preset: str | None = None) -> dict[str, Any]:
    """A foreign checkpoint as a checkpoint of ours, ready for torch.save."""
    raw, averaged, epoch = READERS[fmt](file, ema)
    model_state, model = convert_state(raw, RENAMES[fmt], config)
    ema_state = (model_state if averaged is raw  # one set of weights: stored once
                 else convert_state(averaged, RENAMES[fmt], config)[0])
    meta = {
        "model": "jit",
        "jit": model.config(),
        "num_classes": model.label_emb.num_classes,
        # Li & He: x_t = t * x + (1 - t) * noise, the network predicts x.
        "path": "linear",
        "beta_min": 0.0,
        "target": "x_data",
        "data": data,
        "source": {"format": fmt, "file": Path(file).name, "preset": preset,
                   "ema": ema if averaged is not raw else None},
    }
    return {"model": model_state, "ema": ema_state, "epoch": epoch, "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description="Convert foreign weights into a checkpoint of ours.")
    p.add_argument("format", choices=list(READERS))
    p.add_argument("src", help="the file to convert")
    p.add_argument("--preset", choices=list(JIT_PRESETS), required=True,
                   help="the architecture the weights are for")
    p.add_argument("--ema", choices=["ema1", "ema2"], default="ema1",
                   help="jit-official only: which average becomes our EMA weights. "
                        "ema1 (decay 0.9999) is the one Li & He sample with")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    config = JIT_PRESETS[args.preset]
    if config["img_size"] != 256:
        # Li & He scale the noise by 2 at 512x512 (their --noise_scale), so
        # these weights need a Path whose noise is 2 * N(0, I).
        raise SystemExit(f"{args.preset} was trained with noise scaled by 2; no Path here "
                         "does that yet. The /16 models (256x256) convert as they are.")
    ckpt = convert(args.format, args.src, config, ema=args.ema, preset=args.preset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    n = sum(v.numel() for v in ckpt["model"].values())
    print(f"{args.preset}: {n / 1e6:.2f}M parameters, {ckpt['meta']['num_classes']} classes, "
          f"epoch {ckpt['epoch']} -> {out}")


if __name__ == "__main__":
    main()
