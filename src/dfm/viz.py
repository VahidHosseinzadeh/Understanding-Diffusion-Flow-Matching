"""Plotting. On 2D data these are your actual instrument -- prefer them
over staring at loss numbers.

The single most useful picture is `save_velocity_field`: it shows the
whole learned field at a chosen t, so you can see *where* the model is
wrong (fields pointing into a gap between modes, or collapsing to a
single mode) rather than guessing from sample quality.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: works over SSH on the cluster
import matplotlib.pyplot as plt
import torch


def _prep(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_loss_curve(losses: list[float], path: str | Path, window: int = 50) -> None:
    """Raw loss plus a running mean. Flow matching losses are noisy
    because each step sees a random t; the smoothed line is the one to
    read."""
    p = _prep(path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, lw=0.5, alpha=0.35, color="#888", label="per step")
    if len(losses) >= window:
        smooth = torch.tensor(losses).unfold(0, window, 1).mean(dim=1)
        ax.plot(range(window - 1, len(losses)), smooth, lw=1.6, color="#c44", label=f"mean ({window})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save a batch of images in [-1, 1], shape (B, C, H, W), as a PNG grid."""
    from torchvision.utils import make_grid, save_image

    images = (images.clamp(-1, 1) + 1) / 2
    save_image(make_grid(images, nrow=nrow), str(_prep(path)))


def save_scatter_2d(
    samples: torch.Tensor,
    path: str | Path,
    reference: torch.Tensor | None = None,
    lim: float = 3.0,
) -> None:
    """Generated points, optionally over the true distribution in grey."""
    p = _prep(path)
    s = samples.detach().cpu()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if reference is not None:
        r = reference.detach().cpu()
        ax.scatter(r[:, 0], r[:, 1], s=3, alpha=0.18, color="#bbb", label="data", linewidths=0)
    ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.55, color="#c44", label="samples", linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_velocity_field(
    model,
    path_obj,
    target,
    out_path: str | Path,
    device: torch.device,
    times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    grid: int = 20,
    lim: float = 3.0,
    reference: torch.Tensor | None = None,
) -> None:
    """The learned velocity field on a grid, at several times.

    Read it as: drop a particle anywhere at time t and it moves along
    the arrow. At t near 0 the field should sweep broadly inward from
    everywhere; by t near 1 it should be near-zero on the data manifold
    (nothing left to move) and still pointing inward off it.
    """
    p = _prep(out_path)
    was_training = model.training
    model.eval()

    xs = torch.linspace(-lim, lim, grid)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)

    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 3.2))
    axes = [axes] if len(times) == 1 else list(axes)
    for ax, t_val in zip(axes, times):
        t = torch.full((pts.shape[0],), float(t_val), device=device)
        v = target.to_velocity(path_obj, pts, t, model(pts, t)).cpu()
        if reference is not None:
            r = reference.detach().cpu()
            ax.scatter(r[:, 0], r[:, 1], s=2, alpha=0.15, color="#77a", linewidths=0)
        ax.quiver(
            gx.reshape(-1), gy.reshape(-1), v[:, 0], v[:, 1],
            v.norm(dim=1), cmap="viridis", scale=30, width=0.004,
        )
        ax.set_title(f"t = {t_val:g}", fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    if was_training:
        model.train()


def save_trajectories(
    traj: torch.Tensor,
    path: str | Path,
    n_paths: int = 200,
    lim: float = 3.0,
) -> None:
    """Paths taken from noise (t=0) to data (t=1).

    traj is the (steps+1, B, 2) stack returned by a sampler with
    return_trajectory=True. For rectified flow these should be close to
    straight -- that is the property the method is named for, and seeing
    how straight they actually are is the point of the plot.
    """
    p = _prep(path)
    tr = traj.detach().cpu()[:, :n_paths]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(tr[:, :, 0], tr[:, :, 1], lw=0.4, alpha=0.35, color="#48a")
    ax.scatter(tr[0, :, 0], tr[0, :, 1], s=5, color="#888", label="t=0 (noise)", linewidths=0)
    ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=5, color="#c44", label="t=1 (data)", linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
