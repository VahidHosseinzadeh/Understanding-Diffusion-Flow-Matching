"""A plain MLP for 2D toy data -- the recommended place to start.

On 2D data you can plot the *entire* learned velocity field and the
whole target distribution on one page, so you can see what the model
has learned rather than inferring it from sample quality. Training runs
in seconds on CPU, which means you can afford to be wrong repeatedly.

Same contract as the UNet: forward(x, t) -> tensor shaped like x, with
t a float tensor in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import TimeEmbedding


class MLP(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        hidden: int = 256,
        depth: int = 4,
        time_dim: int = 64,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim, hidden)

        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)
        # Start with a zero velocity field: the model begins by predicting
        # "nothing moves" and learns motion from there, which is a much
        # calmer initialisation than a random field.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.time_emb(t)
        for block in self.blocks:
            h = h + block(h)  # residual: keeps gradients healthy at depth
        return self.out_proj(self.out_norm(h))
