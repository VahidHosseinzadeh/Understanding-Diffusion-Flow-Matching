"""JiT, "Just image Transformers" (Li & He 2025, "Back to Basics: Let
Denoising Generative Models Denoise"): a plain Vision Transformer on
patches of raw pixels.

Cut the image into p x p patches, embed each patch as one token, run a
stack of Transformer blocks, and project each token back to its patch.
That is DiT (Peebles & Xie 2023) moved from a VAE's latents onto pixels,
plus the general-purpose upgrades language models adopted since --
RMSNorm, SwiGLU, rotary positions, qk-norm -- and copies of the class
embedding placed in the token sequence itself.

What makes raw pixels workable is the target, not the architecture. An
ImageNet patch of 16x16x3 is 768 numbers, as many as JiT-B has per
token. Noise and velocity both contain the noise sample, so predicting
either means carrying 768 structureless numbers per token through every
block, while a clean patch lies near a low-dimensional manifold. JiT
therefore predicts x_data and measures the error in velocity space:

    python dfm/train.py --data fashion_mnist --model jit \\
        --target x_data --loss-space velocity

At 28x28 with the default 4x4 patches a token is only 16 numbers against
256 hidden units: the regime of JiT-B/4 on ImageNet 64x64, where all nine
target/loss pairs worked. Large patches bring the bottleneck back
(--patch-size 14 makes every token 196 numbers).

Interface contract, the same as the UNet's: `forward(x, t, y=None)` with
x (B, C, H, W), t (B,) in [0, 1], and y class labels (NULL_LABEL for "no
label") only for a model built with `num_classes`.

Differences from the reference code (github.com/LTH14/JiT). That code
itself, for running Li & He's pretrained weights, is jit_official.py.
  - time goes through `embeddings.TimeEmbedding`, shared with the UNet
    and the MLP, which multiplies t by 1000 first. JiT's code passes t in
    [0, 1] to DiT's embedder unscaled, so its time features vary slowly.
  - the label table starts at zero (`embeddings.LabelEmbedding`).
  - a norm followed by a modulation has no gain of its own: adaLN's
    (1 + scale) already is one, so a second only reparameterises it
    (DiT's LayerNorm drops it too).
  - patches are embedded by a linear map on `patchify`'s vectors instead
    of a stride-p convolution (the same map), attention runs through
    F.scaled_dot_product_attention, and positions are buffers rebuilt
    from the config. No .cuda() and no torch.compile, so it runs on CPU,
    MPS and CUDA alike.
  - no dropout: JiT uses it only in its two largest models.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import NULL_LABEL, LabelEmbedding, TimeEmbedding


# --------------------------------------------------------------------
# patches and positions
# --------------------------------------------------------------------

def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, p*p*C): one row per p x p patch, patches in
    reading order, each flattened pixel by pixel.

    Once a patch is a vector, embedding it is an ordinary linear map, and
    the output layer predicts one such vector per token. Both ends use
    this layout, so "copy the input patch" is the identity on it.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, rows, cols, p, p, C)
    return x.reshape(B, (H // p) * (W // p), p * p * C)


def unpatchify(tokens: torch.Tensor, p: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Inverse of `patchify`: (B, N, p*p*C) -> (B, C, H, W)."""
    B = tokens.shape[0]
    x = tokens.reshape(B, height // p, width // p, p, p, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, rows, p, cols, p)
    return x.reshape(B, channels, height, width)


def _grid(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row and column of every patch, in `patchify` order."""
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    return r.flatten().float(), c.flatten().float()


def sincos_positions(rows: int, cols: int, dim: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position codes, (rows*cols, dim).

    Attention treats its tokens as a set, so without these the network
    could not tell where a patch came from. The 1D codes of Vaswani et al.
    2017, once for the row in half the channels and once for the column
    in the other half, as in MAE and DiT.
    """
    def axis(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = 10000.0 ** (-torch.arange(d // 2, dtype=torch.float64) / (d // 2))
        angle = pos.double()[:, None] * omega[None, :]
        return torch.cat([angle.sin(), angle.cos()], dim=1)

    r, c = _grid(rows, cols)
    return torch.cat([axis(r, dim // 2), axis(c, dim // 2)], dim=1).float()


class PatchEmbed(nn.Module):
    """Patch vector -> token, optionally through a deliberately narrow
    linear layer.

    With `bottleneck`, the map is two linear layers, p*p*C -> bottleneck
    -> width, with nothing between them: a linear map of rank at most
    `bottleneck`. For x_data prediction Li & He found this harmless even
    at 16 dimensions for a 768-number patch, and up to ~1.3 FID better
    across 32-512 -- they read it as manifold learning: a clean patch is
    described by far fewer numbers than it has pixels.
    """

    def __init__(self, patch_dim: int, width: int, bottleneck: int | None = None):
        super().__init__()
        if bottleneck is None:
            self.proj = nn.Linear(patch_dim, width)
        else:
            self.proj = nn.Sequential(
                nn.Linear(patch_dim, bottleneck, bias=False),
                nn.Linear(bottleneck, width),
            )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Turn every consecutive channel pair (a, b) by 90 degrees, to (-b, a)."""
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-b, a), dim=-1).flatten(-2)


class Rope2d(nn.Module):
    """Rotary position embedding on the patch grid (Su et al. 2021,
    "RoFormer"), in the axial 2D form JiT takes from Lightning-DiT.

    A head's channels are taken in pairs, and pair i of a token at grid
    position m is rotated by the angle m * omega_i: the first half of the
    pairs by the token's row, the second half by its column. Rotations
    compose, R_m^T R_n = R_{n-m}, so after rotating queries and keys

        <R_m q, R_n k> = <q, R_{n-m} k>

    and an attention score depends on where two patches are relative to
    each other, not on where they are. JiT keeps the absolute codes
    (`sincos_positions`) on the tokens as well.

    Prefix tokens -- the in-context class tokens -- are not rotated: they
    have no place on the grid.
    """

    def __init__(self, head_dim: int, rows: int, cols: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"2D RoPE needs a head dimension divisible by 4, got {head_dim}")
        n = head_dim // 4  # frequencies per axis, one channel pair each
        omega = theta ** (-torch.arange(n, dtype=torch.float32) / n)
        r, c = _grid(rows, cols)
        angle = torch.cat([r[:, None] * omega, c[:, None] * omega], dim=1)  # (N, head_dim / 2)
        angle = angle.repeat_interleave(2, dim=1)  # both channels of a pair turn together
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(self, x: torch.Tensor, n_prefix: int = 0) -> torch.Tensor:
        """x: (B, heads, n_prefix + N, head_dim), prefix tokens first."""
        cos, sin = self.cos, self.sin
        if n_prefix:
            cos = torch.cat([cos.new_ones(n_prefix, cos.shape[1]), cos])
            sin = torch.cat([sin.new_zeros(n_prefix, sin.shape[1]), sin])
        return x * cos + rotate_pairs(x) * sin


# --------------------------------------------------------------------
# the block
# --------------------------------------------------------------------

class RMSNorm(nn.Module):
    """x / rms(x) (Zhang & Sennrich 2019): LayerNorm without subtracting
    the mean, as in LLaMA-style Transformers.

    `gain=False` drops the learned per-channel gain, for a norm that a
    modulation follows: adaLN's (1 + scale) is already a gain.
    """

    def __init__(self, dim: int, gain: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if gain else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return y if self.weight is None else y * self.weight


class Attention(nn.Module):
    """Multi-head self-attention over all tokens, with qk-norm and RoPE.

    Every token reads every other in every block, so the prediction for a
    patch can depend on the far corner from the first layer on; the UNet
    reaches its one attention layer only after two downsamplings.
    qk-norm (Henry et al. 2020; Dehghani et al. 2023) normalises queries
    and keys per head before the dot product, so attention logits cannot
    grow with the activations and saturate the softmax.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, rope: Rope2d, n_prefix: int) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q = rope(self.q_norm(q), n_prefix)
        k = rope(self.k_norm(k), n_prefix)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    """w3(SiLU(w1 x) * w2 x) (Shazeer 2020): one projection gates the
    other. Three matrices instead of a GELU MLP's two, so the hidden width
    is 2/3 of mlp_ratio * width, which keeps the parameter count."""

    def __init__(self, width: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 / 3 * mlp_ratio * width)
        self.w12 = nn.Linear(width, 2 * hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class JiTBlock(nn.Module):
    """A pre-norm Transformer block conditioned by adaLN-Zero (Peebles &
    Xie 2023).

    The conditioning vector c sets six vectors per block: a shift and a
    scale after each norm, and a gate on each residual branch,

        x <- x + g1 * Attn(norm(x) * (1 + s1) + b1)
        x <- x + g2 * MLP (norm(x) * (1 + s2) + b2)

    The UNet's ResBlock gets only a shift; this adds a scale and a gate.
    The map producing them starts at zero, so every gate is 0 and every
    block is the identity at initialisation: the network starts as a
    stack of no-ops and learns how much each branch should contribute.
    DiT measured this against feeding t and y as extra tokens: FID 19.5
    against 35.2.
    """

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(width, gain=False)
        self.attn = Attention(width, heads)
        self.norm2 = RMSNorm(width, gain=False)
        self.mlp = SwiGLU(width, mlp_ratio)
        self.ada = nn.Linear(width, 6 * width)  # zero-initialised by JiT

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: Rope2d, n_prefix: int = 0) -> torch.Tensor:
        b1, s1, g1, b2, s2, g2 = self.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + s1) + b1, rope, n_prefix)
        return x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)


class FinalLayer(nn.Module):
    """Norm, one last adaptive shift and scale, and a linear map from each
    token back to its p*p*C pixels. Zero-initialised, so a fresh model
    predicts 0 everywhere, as the UNet does."""

    def __init__(self, width: int, patch_dim: int):
        super().__init__()
        self.norm = RMSNorm(width, gain=False)
        self.ada = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, patch_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, s = self.ada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        return self.out(self.norm(x) * (1 + s) + b)


# --------------------------------------------------------------------
# the model
# --------------------------------------------------------------------

class JiT(nn.Module):
    """Patches -> tokens -> `depth` adaLN-Zero blocks -> patches.

    Defaults are sized for Fashion-MNIST: 4x4 patches (49 tokens of 16
    numbers), width 256, 6 blocks, 4 heads of 64 channels -- 7.4M
    parameters, against the UNet's 6.1M. JiT-B is 12 blocks of width 768.

    `in_context_tokens` > 0 adds JiT's second route for the label: that
    many copies of its embedding, each plus a learned vector of its own,
    prepended to the sequence before block `in_context_start` (default
    depth // 3, as JiT-B's 4 of 12) and dropped before the output layer.
    Every patch can then attend to the label directly, on top of adaLN.
    JiT uses 32 next to 256 patches, worth ~1.2 FID on ImageNet. They carry
    the label, so they need `num_classes`.
    """

    def __init__(
        self,
        img_size: int = 28,
        in_channels: int = 1,
        patch_size: int = 4,
        width: int = 256,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int | None = None,
        in_context_tokens: int = 0,
        in_context_start: int | None = None,
        time_freqs: int = 256,
        num_classes: int | None = None,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"patch size {patch_size} does not divide image size {img_size}")
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        if in_context_start is None:
            in_context_start = depth // 3
        if in_context_tokens:
            if num_classes is None:
                raise ValueError("in-context tokens carry the class label; they need num_classes")
            if not 0 <= in_context_start < depth:
                raise ValueError(f"in_context_start must be a block index in [0, {depth})")
        # Everything needed to rebuild the architecture, defaults resolved,
        # so a checkpoint stays loadable if a default here ever changes.
        self._config = dict(
            img_size=img_size, in_channels=in_channels, patch_size=patch_size,
            width=width, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
            bottleneck_dim=bottleneck_dim, in_context_tokens=in_context_tokens,
            in_context_start=in_context_start, time_freqs=time_freqs,
        )
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.in_context_start = in_context_start
        grid = img_size // patch_size
        patch_dim = patch_size * patch_size * in_channels

        # Conditioning: the same two embeddings as the UNet, summed.
        self.time_emb = TimeEmbedding(time_freqs, width)
        self.label_emb = LabelEmbedding(num_classes, width) if num_classes else None
        self.context_pos = (
            nn.Parameter(0.02 * torch.randn(in_context_tokens, width)) if in_context_tokens else None
        )

        self.patch_embed = PatchEmbed(patch_dim, width, bottleneck_dim)
        self.register_buffer("pos", sincos_positions(grid, grid, width), persistent=False)
        self.rope = Rope2d(width // heads, grid, grid)
        self.blocks = nn.ModuleList(JiTBlock(width, heads, mlp_ratio) for _ in range(depth))
        self.final = FinalLayer(width, patch_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """DiT's initialisation, which JiT keeps."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.time_emb.net[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.net[2].weight, std=0.02)
        # adaLN-Zero: every gate, the last modulation and the output start at 0.
        for layer in [b.ada for b in self.blocks] + [self.final.ada, self.final.out]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def config(self) -> dict[str, Any]:
        """Constructor arguments except `num_classes`: what a checkpoint
        stores to rebuild this architecture (`checkpoint.load_checkpoint`)."""
        return dict(self._config)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if (C, H, W) != (self.in_channels, self.img_size, self.img_size):
            raise ValueError(f"JiT built for {self.in_channels}x{self.img_size}x{self.img_size} "
                             f"images, got {C}x{H}x{W}")
        c = self.time_emb(t)
        label = None
        if self.label_emb is not None:
            if y is None:
                y = torch.full((B,), NULL_LABEL, dtype=torch.long, device=x.device)
            label = self.label_emb(y)
            c = c + label
        elif y is not None:
            raise ValueError("labels given to a JiT built without num_classes")

        h = self.patch_embed(patchify(x, self.patch_size)) + self.pos
        n_prefix = 0
        for i, block in enumerate(self.blocks):
            if self.context_pos is not None and i == self.in_context_start:
                h = torch.cat([label[:, None, :] + self.context_pos, h], dim=1)
                n_prefix = self.context_pos.shape[0]
            h = block(h, c, self.rope, n_prefix)
        h = self.final(h[:, n_prefix:], c)
        return unpatchify(h, self.patch_size, C, H, W)
