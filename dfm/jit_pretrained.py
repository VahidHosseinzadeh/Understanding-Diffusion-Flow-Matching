"""Li & He's pretrained JiT models behind this repo's model interface.

Their weights run in their own code, never converted into our JiT
(jit.py): jit_official.py on a CUDA GPU, jit_mac.py anywhere else, the
two differing only in their CUDA calls. This file is ours: it picks
the copy for the device, translates labels, works out which of their six
models a checkpoint holds, and loads it. `checkpoint.load_checkpoint`
calls `load_official` when it meets their checkpoint-last.pth, so
sample.py and every sampler use their models like ours:

    python dfm/sample.py --checkpoint jit-b-16/checkpoint-last.pth --sampler heun \\
        --steps 50 --cfg-scale 3 --guidance-interval 0.1 1 --label 207 360 387 974
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from embeddings import NULL_LABEL

# What tells their six models apart, from the constructors in model_jit.py:
# (depth, hidden size, patch size).
SIGNATURE = {
    "JiT-B/16": (12, 768, 16), "JiT-B/32": (12, 768, 32),
    "JiT-L/16": (24, 1024, 16), "JiT-L/32": (24, 1024, 32),
    "JiT-H/16": (32, 1280, 16), "JiT-H/32": (32, 1280, 32),
}
# Their /16 models take 256x256 images with noise N(0, I), and their /32
# models 512x512 images with noise 2 * N(0, I) (--img_size and
# --noise_scale in their README).
IMG_SIZE = {name: 256 if name.endswith("/16") else 512 for name in SIGNATURE}
NOISE_SCALE = {name: 1.0 if name.endswith("/16") else 2.0 for name in SIGNATURE}


def their_code(device: torch.device) -> ModuleType:
    """Their code as written for a CUDA device; the .cuda()-free copy otherwise."""
    return importlib.import_module("jit_official" if device.type == "cuda" else "jit_mac")


class OfficialJiT(nn.Module):
    """Their JiT with this repo's model contract, forward(x, t, y=None).

    The network sits at `.net`, where their Denoiser keeps it, so the
    state dicts in their checkpoints load as they are. Only the label
    needs translating: their "no label" is the last row of the label
    table, index num_classes, and ours is NULL_LABEL. Time already
    agrees: they also interpolate x_t = t * x + (1 - t) * noise, with
    t = 0 noise, and pass t in [0, 1].
    """

    def __init__(self, name: str, num_classes: int = 1000, code: ModuleType | None = None):
        super().__init__()
        code = code or their_code(torch.device("cuda"))
        self.name = name
        self.img_size = IMG_SIZE[name]
        self.net = code.JiT_models[name](input_size=self.img_size, in_channels=3, num_classes=num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        null = self.net.num_classes
        if y is None:
            y = torch.full((x.shape[0],), null, dtype=torch.long, device=x.device)
        else:
            # Checked because their embedding table does not: on MPS an
            # out-of-range row is read as garbage memory, not an error.
            if ((y < 0) & (y != NULL_LABEL)).any() or (y >= null).any():
                raise ValueError(f"labels must be in [0, {null}) or NULL_LABEL")
            y = torch.where(y == NULL_LABEL, null, y)
        return self.net(x, t, y)


def identify(state: dict[str, torch.Tensor]) -> str:
    """Which of their models a state dict is for. Their checkpoint files
    do not say, but depth, width and patch size, all readable off the
    tensors, tell the six apart. The strict load that follows checks
    every other name and shape."""
    depth = len({k.split(".")[2] for k in state if k.startswith("net.blocks.")})
    width = state["net.pos_embed"].shape[-1]
    patch = state["net.x_embedder.proj1.weight"].shape[-1]
    for name, signature in SIGNATURE.items():
        if signature == (depth, width, patch):
            return name
    raise ValueError(f"no model of Li & He's has depth {depth}, width {width}, patch {patch}")


def load_official(ckpt: dict[str, Any], device: torch.device,
                  use_ema: bool = True) -> tuple[OfficialJiT, dict[str, Any]]:
    """Their checkpoint-last.pth, as torch.load returns it -> (model, meta).

    The file holds their Denoiser's raw weights ('model') and two EMAs.
    'model_ema1' (decay 0.9999) is the one they sample with, so it stands
    in for our EMA. `meta` records what the weights cannot show: Li & He
    train on the straight path and the network predicts x_data.
    """
    state = ckpt["model_ema1" if use_ema else "model"]
    name = identify(state)
    if NOISE_SCALE[name] != 1.0:
        raise SystemExit(f"{name} was trained with noise of standard deviation "
                         f"{NOISE_SCALE[name]:g} (their --noise_scale), which no Path here provides yet")
    num_classes = state["net.y_embedder.embedding_table.weight"].shape[0] - 1
    model = OfficialJiT(name, num_classes, their_code(device))
    model.load_state_dict(state)  # strict: every name and shape must be theirs
    meta = {
        "model": "jit_official", "jit_official": name, "img_size": model.img_size,
        "num_classes": num_classes, "path": "linear", "beta_min": 0.0, "target": "x_data",
        "data": "imagenet",
    }
    return model, meta
