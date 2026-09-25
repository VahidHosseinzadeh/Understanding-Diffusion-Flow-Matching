"""Sample-quality metrics for image models: FID, KID, IS, precision/recall.

The numbers come from torch-fidelity, the package JiT (Li & He 2025) and
most recent diffusion papers compute them with. That is deliberate: FID
depends on details nobody should reimplement -- the TF1-compatible
Inception-v3 weights, its bilinear resize to 299x299, the covariance
estimator -- and a reimplementation that differs in any of them yields
numbers that look like FIDs and compare with nothing. (JiT pins a fork
that reads precomputed statistics from an .npz; upstream torch-fidelity,
used here, gets the same effect by caching them.)

What torch-fidelity cannot know, and this module owns:

  - how model output maps to pixels. Samples live in [-1, 1], the
    training normalisation; real images are stored as uint8. Both must
    reach the feature extractor quantised identically, or FID partly
    measures the conversion instead of the model.
  - drawing N samples in batches from any (model, path, target, sampler),
    from a fixed seed.
  - caching the real images' statistics, but never the generated ones.

The metrics, each comparing generated samples against real ones:

  fid  Frechet Inception Distance (Heusel et al. 2017). Lower is better.
       Biased upward at small N: compare runs only at equal N.
  kid  Kernel Inception Distance (Binkowski et al. 2018). Unbiased, so
       steadier than FID at a few thousand samples. Lower is better.
  isc  Inception Score (Salimans et al. 2016). Its classes are ImageNet's,
       which do not describe Fashion-MNIST: a relative signal only.
  prc  Precision and recall (Kynkaanniemi et al. 2019): fidelity and
       coverage separately, the two failures FID folds into one number.
       Uses VGG-16 features, so it costs a second download and pass.

torch-fidelity is imported lazily, so training and the tests never need it.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import get_fashion_mnist
from paths import Path
from targets import Target

METRICS = ("fid", "isc", "kid", "prc")


def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
    """Images in [-1, 1], (B, C, H, W) -> uint8 (B, 3, H, W).

    Exactly inverts the training normalisation, ToTensor then
    Normalize(0.5, 0.5): a stored pixel p becomes 2p/255 - 1 in training
    and comes back as p here. Clamping first matters -- samples overshoot
    [-1, 1], and an unclamped -1.2 wraps around in the uint8 cast to a
    near-white pixel. Grey is repeated to three channels because the
    feature extractors expect RGB.
    """
    x = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x


class UInt8Images(Dataset):
    """A fixed set of uint8 (N, 3, H, W) images, in the form torch-fidelity
    takes: each item must *be* the image tensor, which rules out
    TensorDataset and its 1-tuples."""

    def __init__(self, images: torch.Tensor):
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.images[i]


def fashion_mnist_uint8(train: bool = True) -> torch.Tensor:
    """The real images as uint8 (N, 3, 28, 28), straight from the stored
    pixels -- so the reference involves no float round trip at all."""
    return get_fashion_mnist(train=train).data.unsqueeze(1).expand(-1, 3, -1, -1)


@torch.no_grad()
def generate_images(
    model,
    path: Path,
    target: Target,
    sampler: Callable,
    n: int,
    sample_shape: tuple[int, ...],
    device: torch.device,
    steps: int,
    batch_size: int = 500,
    seed: int = 0,
    progress: bool = True,
) -> torch.Tensor:
    """Draw `n` samples in batches; returns uint8 (n, 3, H, W) on the CPU.

    Seeded here rather than by the caller, so the deterministic samplers
    decode exactly the same starting noise at every step count: the gap
    between two settings is then the solver's doing, not a new draw.
    Quantised batch by batch, which also keeps 50k samples small.
    """
    torch.manual_seed(seed)
    batches = []
    for start in tqdm(range(0, n, batch_size), desc="generating", disable=not progress):
        shape = (min(batch_size, n - start), *sample_shape)
        x = sampler(model, path, target, shape, device, steps=steps, progress=False)
        batches.append(to_uint8_rgb(x).cpu())
    return torch.cat(batches)


def image_metrics(
    samples: torch.Tensor,
    reference: torch.Tensor,
    reference_name: str,
    metrics: Sequence[str] = ("fid", "isc", "kid"),
    cache_dir: str | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """Score uint8 (N, 3, H, W) `samples` against `reference` images.

    The reference's features and statistics are cached under
    `reference_name`, so the pass over the real images runs once per
    dataset split, not once per evaluation. The samples get no cache
    name, deliberately: cached, every later checkpoint would silently be
    scored with the first one's features.

    Returns torch-fidelity's own keys (`frechet_inception_distance`,
    `inception_score_mean`, `kernel_inception_distance_mean`,
    `precision`, ...), the same names JiT's code reads.
    """
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {METRICS}")
    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError("torch-fidelity is not installed. Run: pip install torch-fidelity") from exc

    scores = torch_fidelity.calculate_metrics(
        input1=UInt8Images(samples),
        input2=UInt8Images(reference),
        input2_cache_name=reference_name,
        cache_root=cache_dir,
        cuda=torch.cuda.is_available(),  # CUDA or CPU: torch-fidelity has no MPS path
        # KID averages over random subsets of this size, which can be no
        # larger than either set.
        kid_subset_size=min(1000, len(samples), len(reference)),
        # Misnamed but wanted, and it changes no number. It turns off loader
        # workers, which buy nothing for images already in memory and crash
        # on macOS when the caller has no `if __name__ == "__main__"`; and it
        # makes precision/recall work in blocks instead of materialising the
        # reference-vs-reference distance matrix (60k x 60k floats = 14 GB).
        save_cpu_ram=True,
        verbose=verbose,
        **{m: m in metrics for m in METRICS},
    )
    return {k: float(v) for k, v in scores.items()}
