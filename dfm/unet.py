"""A UNet for image data (Fashion-MNIST at 28x28).

Interface contract: `UNet.forward(x, t)` where
  x: (B, C, H, W) float tensor, the interpolated input x_t
  t: (B,) float tensor in [0, 1]

Identical contract to `dfm.mlp.MLP`, so the model is an independent
choice alongside path/target/sampler -- swapping it never touches the
process code. Time conditioning is shared with the MLP via
`dfm.embeddings`.

Kept small and readable (no attention above the lowest resolution) so
it trains at a reasonable pace on CPU/MPS. Bump `base_channels` or
`channel_mults` for more capacity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimeEmbedding


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with the time embedding injected as a
    per-channel bias (FiLM-lite) after the first conv."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Plain single-head self-attention over spatial positions. Only
    used at the lowest resolution (7x7 for Fashion-MNIST) where the
    O(N^2) cost over N=H*W tokens is trivial."""

    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        k = k.reshape(B, C, H * W)                    # (B, C, N)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)   # (B, N, C)
        attn = torch.softmax(q @ k / math.sqrt(C), dim=-1)  # (B, N, N)
        out = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        use_attention_at_lowest_res: bool = True,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = TimeEmbedding(base_channels, time_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- down path ---
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        chs = [ch]  # track channel counts for skip connections
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim))
                ch = out_ch
                chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                chs.append(ch)
            else:
                self.downsamples.append(None)

        # --- bottleneck ---
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention2d(ch) if use_attention_at_lowest_res else nn.Identity()
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # --- up path (mirrors down path, consuming skip connections) ---
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim))
                ch = out_ch
            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)  # common trick: start near identity/zero output
        nn.init.zeros_(self.out_conv.bias)

        self._channel_mults = channel_mults
        self._num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = [h]

        idx = 0
        for i, mult in enumerate(self._channel_mults):
            for _ in range(self._num_res_blocks):
                h = self.down_blocks[idx](h, t_emb)
                idx += 1
                skips.append(h)
            down = self.downsamples[i]
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        idx = 0
        for up_idx, (i, mult) in enumerate(reversed(list(enumerate(self._channel_mults)))):
            for _ in range(self._num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[idx](h, t_emb)
                idx += 1
            # NOTE: self.upsamples was built by *appending* during this same
            # reversed loop at construction time, so it must be indexed
            # positionally here (up_idx), not by the level index i.
            up = self.upsamples[up_idx]
            if up is not None:
                h = up(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
