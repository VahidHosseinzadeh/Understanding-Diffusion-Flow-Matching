"""Conditioning, shared by every model in this package: time, and labels.

Both models take t as a float tensor in [0, 1] -- never a raw integer
timestep. Discrete-time methods normalise before calling the model, so
the network never has to know how many steps a schedule was defined
with, and a model trained under one discretisation can be sampled under
another.

A class label enters through the same door as time: its embedding is
added to the time embedding, and every block conditions on the sum.
Nothing here knows what the network predicts, so conditioning works the
same for every target.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Transformer-style sinusoidal features for t in [0, 1].

    The rescale by 1000 is convention, not necessity: diffusion papers
    embed an integer step in [0, 1000), so multiplying keeps the
    frequency band in the range those hyperparameters were tuned for.
    Without it the low frequencies are nearly constant across [0, 1] and
    the network struggles to tell nearby times apart.
    """
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """sinusoidal features -> MLP -> a vector each block can condition on."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.dim))


# "No label". Classifier-free guidance needs the unconditional field from
# the same network as the conditional ones, so "no label" is one more
# input value rather than a different model (Ho & Salimans 2022).
NULL_LABEL = -1


class LabelEmbedding(nn.Module):
    """One learned vector per class, plus one for NULL_LABEL.

    Starts at zero, so an untrained network ignores the label and learns
    how much it matters. Zero rows are not stuck together: each row's
    gradient comes only from the samples carrying that label.
    """

    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.table = nn.Embedding(num_classes + 1, dim)  # last row is NULL_LABEL
        nn.init.zeros_(self.table.weight)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Checked here because nothing downstream does: on MPS an
        # out-of-range row is read as garbage memory, not an error.
        if ((y < 0) & (y != NULL_LABEL)).any() or (y >= self.num_classes).any():
            raise ValueError(f"labels must be in [0, {self.num_classes}) or NULL_LABEL")
        return self.table(torch.where(y == NULL_LABEL, self.num_classes, y))
