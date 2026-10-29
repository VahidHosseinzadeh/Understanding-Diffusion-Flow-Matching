"""JiT (Li & He 2025), checked piece by piece: the patch layout, the
rotary positions, adaLN-Zero's identity start, the label's two routes,
and the checkpoint round trip. Each test checks a property the design
relies on, not just a shape."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from checkpoint import load_checkpoint
from embeddings import NULL_LABEL
from jit import JiT, PatchEmbed, Rope2d, patchify, unpatchify
from losses import interpolant_loss
from paths import LinearPath
from targets import DataTarget

# 8x8 images in 2x2 patches: 16 tokens of 4 numbers, width 32, 2 heads of 16.
SMALL = dict(img_size=8, patch_size=2, width=32, depth=3, heads=2)


def _randomise(model: nn.Module) -> nn.Module:
    """adaLN-Zero and the zero output layer make a fresh JiT answer 0 for
    every input: perturb every parameter first."""
    torch.manual_seed(0)
    with torch.no_grad():
        for q in model.parameters():
            q.add_(0.1 * torch.randn_like(q))
    return model.eval()


# --------------------------------------------------------------------
# patches and positions
# --------------------------------------------------------------------

def test_patchify_round_trips_and_keeps_each_patch_together():
    x = torch.randn(2, 3, 8, 12)
    tokens = patchify(x, 4)
    assert tokens.shape == (2, 6, 4 * 4 * 3)
    assert torch.equal(unpatchify(tokens, 4, 3, 8, 12), x)
    # Reading order, three patches per row: token 4 is row 1, column 1.
    patch = x[:, :, 4:8, 4:8]
    assert torch.equal(tokens[:, 4], patch.permute(0, 2, 3, 1).reshape(2, -1))


def test_rope_scores_depend_only_on_the_offset_between_patches():
    """<R_m q, R_n k> = <q, R_(n-m) k>: the same offset gives the same
    score wherever the pair sits on the grid."""
    rope = Rope2d(head_dim=8, rows=4, cols=4)
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    # The same q and k at every position, so scores differ only by position.
    rq = rope(q.expand(1, 1, 16, 8))[0, 0]
    rk = rope(k.expand(1, 1, 16, 8))[0, 0]
    scores = rq @ rk.T  # scores[m, n] = <R_m q, R_n k>

    def at(row, col):
        return 4 * row + col

    ref = scores[at(0, 0), at(1, 2)]  # one row down, two columns right
    for r, c in [(1, 1), (2, 0), (2, 1)]:
        assert torch.allclose(scores[at(r, c), at(r + 1, c + 2)], ref, atol=1e-5)
    assert not torch.allclose(scores[at(0, 0), at(2, 1)], ref, atol=1e-3)
    # Rotations keep lengths.
    assert torch.allclose(rq.norm(dim=-1), q.norm().expand(16), atol=1e-5)


def test_rope_leaves_the_class_tokens_unrotated():
    rope = Rope2d(head_dim=8, rows=2, cols=2)
    x = torch.randn(1, 2, 3 + 4, 8)  # 3 prefix tokens, then the 2x2 grid
    out = rope(x, n_prefix=3)
    assert torch.equal(out[:, :, :3], x[:, :, :3])
    assert torch.allclose(out[:, :, 3:], rope(x[:, :, 3:]))


def test_bottleneck_embedding_is_a_low_rank_linear_map():
    """JiT's bottleneck: two linear layers through `bottleneck` numbers,
    nothing in between, so the map has rank `bottleneck`."""
    torch.manual_seed(0)
    narrow, plain = PatchEmbed(48, 64, bottleneck=8), PatchEmbed(48, 64)
    for embed, rank in [(narrow, 8), (plain, 48)]:
        matrix = embed(torch.eye(48)) - embed(torch.zeros(1, 48))  # image of each basis vector
        assert torch.linalg.matrix_rank(matrix) == rank


# --------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------

def test_adaln_zero_starts_every_block_as_the_identity():
    model = JiT(**SMALL, num_classes=4)
    x, t = torch.randn(3, 1, 8, 8), torch.rand(3)
    assert torch.equal(model(x, t, torch.tensor([0, 1, 2])), torch.zeros_like(x))
    torch.manual_seed(0)
    h, c = torch.randn(3, 16, 32), torch.randn(3, 32)
    for block in model.blocks:
        assert torch.equal(block(h, c, model.rope), h)


def test_zero_initialisation_does_not_block_any_parameter():
    """With the output layer and every gate at exactly 0, the first
    gradient reaches only the output layer. Each step opens the next layer
    of zeros, and after three every parameter is learning."""
    torch.manual_seed(0)
    model = JiT(**SMALL, num_classes=3, in_context_tokens=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 1, 8, 8)
    y = torch.tensor([0, 1, 2, NULL_LABEL] * 2)  # every row of the label table

    def grads() -> dict[str, float]:
        opt.zero_grad()
        interpolant_loss(model, x, LinearPath(), DataTarget(), labels=y).backward()
        return {n: q.grad.abs().max().item() for n, q in model.named_parameters()}

    first = grads()
    assert first["final.out.weight"] > 0
    assert first["blocks.0.attn.qkv.weight"] == 0 and first["patch_embed.proj.weight"] == 0
    for _ in range(3):
        opt.step()
        last = grads()
    assert all(g > 0 for g in last.values()), [n for n, g in last.items() if g == 0]


def test_one_block_already_sees_the_whole_image():
    """Attention mixes every token in every block: moving one corner
    patch moves the prediction in every patch, with a single block. (One
    3x3 convolution would move only the corner's neighbours.)"""
    model = _randomise(JiT(**{**SMALL, "depth": 1}))
    x, t = torch.randn(1, 1, 8, 8), torch.rand(1)
    moved = x.clone()
    moved[..., :2, :2] += 1.0
    change = patchify(model(moved, t) - model(x, t), 2).abs().amax(dim=-1)
    assert (change > 1e-6).all()


# --------------------------------------------------------------------
# labels
# --------------------------------------------------------------------

def test_conditional_jit_without_labels_is_the_null_label():
    """Samplers and diagnostics call model(x, t); on a conditional model
    that must be the unconditional prediction, exactly."""
    model = _randomise(JiT(**SMALL, num_classes=4, in_context_tokens=2))
    x, t = torch.randn(3, 1, 8, 8), torch.rand(3)
    null = torch.full((3,), NULL_LABEL)
    assert torch.equal(model(x, t), model(x, t, null))
    assert not torch.allclose(model(x, t, torch.tensor([0, 1, 2])), model(x, t, null))


def test_unconditional_jit_has_no_label_parameters_and_refuses_labels():
    model = JiT(**SMALL)
    assert not any("label" in k or "context" in k for k in model.state_dict())
    with pytest.raises(ValueError, match="num_classes"):
        model(torch.randn(2, 1, 8, 8), torch.rand(2), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="num_classes"):
        JiT(**SMALL, in_context_tokens=4)  # the class tokens would carry nothing


def test_class_tokens_join_at_their_block_and_leave_before_the_output():
    model = JiT(**SMALL, num_classes=4, in_context_tokens=5, in_context_start=1)
    lengths = []
    for block in model.blocks:
        block.register_forward_hook(lambda m, inputs, out: lengths.append(out.shape[1]))
    out = model(torch.randn(2, 1, 8, 8), torch.rand(2), torch.tensor([0, 3]))
    assert lengths == [16, 21, 21]
    assert out.shape == (2, 1, 8, 8)


# --------------------------------------------------------------------
# loading
# --------------------------------------------------------------------

def test_checkpoint_rebuilds_the_same_jit(tmp_path):
    model = _randomise(JiT(**SMALL, num_classes=4, in_context_tokens=2, bottleneck_dim=3))
    meta = {"path": "linear", "target": "x_data", "model": "jit", "data": "fashion_mnist",
            "num_classes": 4, "jit": model.config()}
    file = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "meta": meta}, file)
    loaded = load_checkpoint(str(file), torch.device("cpu")).model
    x, t, y = torch.randn(2, 1, 8, 8), torch.rand(2), torch.tensor([1, NULL_LABEL])
    assert torch.equal(loaded(x, t, y), model(x, t, y))
