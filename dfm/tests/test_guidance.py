"""Class conditioning and classifier-free guidance.

The guided field is checked against its formula on a network whose
conditional and unconditional outputs are known in closed form, and the
claim that guidance may combine raw predictions of any target is checked
numerically for every target. Label dropout is checked for the property
the unconditional field depends on: the dropped samples are a random
sample of the data.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from checkpoint import load_checkpoint
from embeddings import NULL_LABEL, LabelEmbedding
from guidance import ClassifierFreeGuidance, balanced_labels, network_passes
from losses import drop_labels, interpolant_loss
from metrics import generate_images
from mlp import MLP
from paths import LinearPath
from samplers import euler, heun
from targets import TARGETS, DataTarget, VelocityTarget
from trainer import Trainer, TrainConfig
from unet import UNet


class Affine(nn.Module):
    """f(x, t, y) = (y + 2) x + t, so NULL_LABEL (-1) gives x + t: every
    output is known in closed form."""

    def forward(self, x, t, y):
        return (y.float() + 2).reshape(-1, 1) * x + t.reshape(-1, 1)


def _randomise(model: nn.Module) -> nn.Module:
    """The UNet zero-initialises its output conv and label table, so a fresh
    one answers 0 whatever its inputs: perturb every parameter first."""
    torch.manual_seed(0)
    with torch.no_grad():
        for q in model.parameters():
            q.add_(0.1 * torch.randn_like(q))
    return model.eval()


# --------------------------------------------------------------------
# the network
# --------------------------------------------------------------------

def test_label_embedding_gives_null_its_own_row():
    emb = LabelEmbedding(num_classes=3, dim=4)
    with torch.no_grad():
        emb.table.weight.copy_(torch.arange(16.0).reshape(4, 4))
    out = emb(torch.tensor([0, 2, NULL_LABEL]))
    assert torch.equal(out[0], emb.table.weight[0])
    assert torch.equal(out[1], emb.table.weight[2])
    assert torch.equal(out[2], emb.table.weight[3])  # the extra row, not row -1 = row 3 by accident
    assert emb.table.weight.shape[0] == 4


@pytest.mark.parametrize("bad", [3, -2])
def test_label_embedding_refuses_unknown_labels(bad):
    """On MPS an out-of-range index reads garbage instead of failing."""
    with pytest.raises(ValueError, match="labels"):
        LabelEmbedding(num_classes=3, dim=4)(torch.tensor([0, bad]))


def test_unconditional_unet_is_unchanged_and_refuses_labels():
    """Old checkpoints must still load: no new parameters without num_classes."""
    model = UNet(base_channels=8, channel_mults=(1,), num_res_blocks=1)
    assert not any("label" in k for k in model.state_dict())
    with pytest.raises(ValueError, match="num_classes"):
        model(torch.randn(2, 1, 8, 8), torch.rand(2), torch.tensor([0, 1]))


def test_conditional_unet_without_labels_is_the_null_label():
    """Samplers and diagnostics call model(x, t); on a conditional model
    that must be the unconditional prediction, exactly."""
    model = _randomise(UNet(base_channels=8, channel_mults=(1,), num_res_blocks=1, num_classes=10))
    x, t = torch.randn(3, 1, 8, 8), torch.rand(3)
    null = torch.full((3,), NULL_LABEL)
    assert torch.equal(model(x, t), model(x, t, null))
    assert not torch.allclose(model(x, t, torch.tensor([0, 1, 2])), model(x, t, null))


# --------------------------------------------------------------------
# the guided field
# --------------------------------------------------------------------

def test_guided_prediction_is_the_cfg_formula():
    x, t, y = torch.randn(4, 3), torch.rand(4), torch.tensor([0, 1, 2, 3])
    cond = (y.float() + 2).reshape(-1, 1) * x + t.reshape(-1, 1)
    uncond = x + t.reshape(-1, 1)
    for w in (0.5, 2.0, 4.0):
        g = ClassifierFreeGuidance(Affine(), y, scale=w)
        assert torch.allclose(g(x, t), (1 - w) * uncond + w * cond, atol=1e-6)
        assert g.network_passes == 2  # one call on a doubled batch


@pytest.mark.parametrize("w, which", [(1.0, "cond"), (0.0, "uncond")])
def test_scale_one_and_zero_need_one_pass(w, which):
    x, t, y = torch.randn(4, 3), torch.rand(4), torch.tensor([0, 1, 2, 3])
    expected = {"cond": (y.float() + 2).reshape(-1, 1) * x, "uncond": x}[which] + t.reshape(-1, 1)
    g = ClassifierFreeGuidance(Affine(), y, scale=w)
    assert torch.allclose(g(x, t), expected)
    assert g.network_passes == 1


def test_guidance_interval_guides_only_inside_it():
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 2, 3])
    t = torch.tensor([0.1, 0.4, 0.6, 0.9])
    w = 3.0
    cond = (y.float() + 2).reshape(-1, 1) * x + t.reshape(-1, 1)
    guided = (1 - w) * (x + t.reshape(-1, 1)) + w * cond
    g = ClassifierFreeGuidance(Affine(), y, scale=w, interval=(0.3, 0.7))
    out = g(x, t)
    inside = torch.tensor([False, True, True, False])
    assert torch.allclose(out[inside], guided[inside], atol=1e-6)
    assert torch.allclose(out[~inside], cond[~inside])

    g = ClassifierFreeGuidance(Affine(), y, scale=w, interval=(0.95, 1.0))
    assert torch.allclose(g(x, t), cond) and g.network_passes == 1  # nobody inside: no extra pass


@pytest.mark.parametrize("name", list(TARGETS))
def test_guidance_commutes_with_conversion_for_every_target(name):
    """Guiding raw predictions and then converting equals converting and
    then guiding, because to_velocity is affine in the prediction and the
    weights (1 - w) + w sum to one. This is what lets guidance wrap the
    network without knowing the target."""
    target, path = TARGETS[name](), LinearPath()
    torch.manual_seed(0)
    x, t = torch.randn(8, 5), torch.empty(8).uniform_(0.1, 0.9)
    p_null, p_y, w = torch.randn(8, 5), torch.randn(8, 5), 3.0

    def to_v(pred):
        return target.to_velocity(path, x, t, pred)

    assert torch.allclose(to_v((1 - w) * p_null + w * p_y),
                          (1 - w) * to_v(p_null) + w * to_v(p_y), atol=1e-5)
    if name != "velocity":
        # The condition is load-bearing: weights that do not sum to one
        # count the x-dependent part of the conversion the wrong number of
        # times. (Velocity's conversion has no such part.)
        assert not torch.allclose(to_v(p_null + w * p_y), to_v(p_null) + w * to_v(p_y), atol=1e-3)


# --------------------------------------------------------------------
# what guidance costs
# --------------------------------------------------------------------

class Counting(nn.Module):
    """Counts network work in units of the sampler's batch size."""

    def __init__(self, batch: int):
        super().__init__()
        self.batch, self.passes = batch, 0

    def forward(self, x, t, y):
        self.passes += x.shape[0] // self.batch
        return torch.zeros_like(x)


@pytest.mark.parametrize("sampler, steps, w, interval, expected", [
    (euler, 10, 1.0, (0.0, 1.0), 10),
    (euler, 10, 2.0, (0.0, 1.0), 20),
    (heun, 5, 2.0, (0.0, 1.0), 18),     # 2N - 1 field evaluations, each doubled
    (euler, 10, 2.0, (0.45, 1.0), 15),  # t = .5,...,.9 guided; t = 0,...,.4 not
])
def test_network_passes_counts_what_the_network_does(sampler, steps, w, interval, expected):
    assert network_passes(sampler, LinearPath(), VelocityTarget(), (3,), steps, w, interval) == expected

    counting = Counting(batch=4)
    g = ClassifierFreeGuidance(counting, torch.zeros(4, dtype=torch.long), w, interval)
    sampler(g, LinearPath(), VelocityTarget(), (4, 3), torch.device("cpu"), steps=steps, progress=False)
    assert counting.passes == g.network_passes == expected


def test_network_passes_leaves_the_rng_alone():
    from samplers import euler_maruyama
    torch.manual_seed(0)
    network_passes(euler_maruyama, LinearPath(), VelocityTarget(), (3,), 5, 2.0)
    after = torch.rand(1)
    torch.manual_seed(0)
    assert torch.equal(after, torch.rand(1))


# --------------------------------------------------------------------
# training with label dropout
# --------------------------------------------------------------------

class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))
        self.seen: list[torch.Tensor] = []
        self.arity: list[int] = []

    def forward(self, x, t, *y):
        self.arity.append(2 + len(y))
        if y:
            self.seen.append(y[0])
        return self.w * x


def test_label_dropout_drops_at_random():
    """The dropped samples must be a random sample of the data: then the
    loss on them is exactly the unconditional loss, and the network's NULL
    output converges to the unconditional prediction (Ho & Salimans 2022).
    So: the drop rate is p in every class, and kept labels are untouched."""
    torch.manual_seed(0)
    n, p = 50_000, 0.3
    labels = balanced_labels(n, 10)
    x_data = labels.float().reshape(-1, 1, 1).expand(-1, 2, 2)  # class k -> image of k's
    rec = Recorder()
    interpolant_loss(rec, x_data, LinearPath(), DataTarget(), labels=labels, label_dropout=p)
    seen = rec.seen[0]

    dropped = seen == NULL_LABEL
    assert torch.equal(seen[~dropped], labels[~dropped])
    assert dropped.float().mean().item() == pytest.approx(p, abs=0.01)
    for k in range(10):
        assert dropped[labels == k].float().mean().item() == pytest.approx(p, abs=0.025)
    # What the NULL row is trained towards: the mean target over all data.
    assert x_data[dropped].mean().item() == pytest.approx(x_data.mean().item(), abs=0.05)


def test_label_dropout_extremes():
    labels = torch.tensor([3, 1, 4, 1, 5])
    assert torch.equal(drop_labels(labels, 0.0), labels)
    assert (drop_labels(labels, 1.0) == NULL_LABEL).all()


def test_unconditional_models_never_see_a_third_argument():
    """Without labels the model is called as model(x_t, t), so the MLP and
    every unconditional checkpoint keep working."""
    rec = Recorder()
    interpolant_loss(rec, torch.randn(4, 2), LinearPath(), VelocityTarget())
    assert rec.arity == [2]
    interpolant_loss(MLP(dim=2, hidden=8, depth=1), torch.randn(4, 2), LinearPath(), VelocityTarget())


def test_trainer_hands_the_loss_the_labels(tmp_path):
    """The trainer used to keep only batch[0], silently dropping labels."""
    got = []

    def loss_fn(model, batch):
        got.append(batch)
        return interpolant_loss(model, batch[0], LinearPath(), VelocityTarget())

    data = torch.utils.data.TensorDataset(torch.randn(8, 2), torch.arange(8))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    Trainer(MLP(dim=2, hidden=8, depth=1), loss_fn, torch.device("cpu"),
            TrainConfig(epochs=1, out_dir=str(tmp_path))).fit(loader)
    assert len(got[0]) == 2 and torch.equal(got[0][1], torch.arange(4))


# --------------------------------------------------------------------
# loading and generating
# --------------------------------------------------------------------

def test_checkpoint_rebuilds_a_conditional_unet(tmp_path):
    model = _randomise(UNet(base_channels=64, num_classes=10))
    meta = {"path": "linear", "target": "velocity", "model": "unet", "data": "fashion_mnist",
            "base_channels": 64, "num_classes": 10}
    file = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "meta": meta}, file)
    loaded = load_checkpoint(str(file), torch.device("cpu")).model
    assert loaded.label_emb is not None
    assert torch.equal(loaded.label_emb.table.weight, model.label_emb.table.weight)


def test_generate_images_hands_each_batch_its_own_labels():
    seen = []

    class Spy(nn.Module):
        def forward(self, x, t, y):
            seen.append(y[: x.shape[0] // 2].clone())  # the conditional half
            return torch.zeros_like(x)

    labels = torch.arange(10)
    generate_images(Spy(), LinearPath(), VelocityTarget(), euler, n=10, sample_shape=(1, 2, 2),
                    device=torch.device("cpu"), steps=1, batch_size=4, progress=False,
                    labels=labels, guidance_scale=2.0)
    assert torch.equal(torch.cat(seen), labels)  # batches of 4, 4, 2, in order
