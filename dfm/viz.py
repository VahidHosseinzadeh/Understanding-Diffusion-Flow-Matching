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


def save_loss_vs_time(
    centres: torch.Tensor,
    means: torch.Tensor,
    counts: torch.Tensor,
    path: str | Path,
) -> None:
    """Per-timestep loss profile. Flat is healthy.

    Empty bins are NaN and matplotlib leaves them as gaps, which is the
    honest rendering -- a bin nothing landed in is missing data, not zero
    loss. The count bars underneath show where the t-distribution
    actually put its samples, so a spike can be read as "hard here"
    rather than "barely sampled here".
    """
    p = _prep(path)
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(centres, means, marker="o", ms=3, lw=1.4, color="#c44")
    ax.set_ylabel("mean loss")
    ax.set_title("loss vs t   (t=0 noise → t=1 data)", fontsize=9)
    ax.grid(alpha=0.2)

    ax_n.bar(centres, counts, width=0.9 / max(len(centres), 1), color="#aaa")
    ax_n.set_ylabel("samples")
    ax_n.set_xlabel("t")
    ax_n.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def save_straightness_hist(values: torch.Tensor, path: str | Path) -> None:
    """Distribution of per-sample trajectory straightness, in (0, 1]."""
    p = _prep(path)
    v = values.detach().cpu().flatten()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(v.numpy(), bins=40, range=(0, 1), color="#48a", alpha=0.85)
    ax.axvline(v.mean().item(), color="#c44", lw=1.6,
               label=f"mean {v.mean().item():.4f}")
    ax.set_xlabel("straightness   ||x1-x0|| / path length     (1.0 = perfectly straight)")
    ax.set_ylabel("samples")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def save_velocity_norm_profile(
    times: torch.Tensor,
    mean_norms: torch.Tensor,
    max_norms: torch.Tensor,
    path: str | Path,
) -> None:
    """||u(x_t, t)|| across t, log-scaled.

    Log y is not optional here: a boundary explosion spans orders of
    magnitude, and on a linear axis it flattens everything else into the
    floor. A velocity-target run should read nearly flat; noise- and
    x_data-prediction should climb steeply at opposite ends.
    """
    p = _prep(path)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(times, mean_norms, lw=1.6, color="#48a", label="mean")
    ax.plot(times, max_norms, lw=1.2, color="#c44", ls="--", label="max")
    ax.set_yscale("log")
    ax.set_xlabel("t   (0 = noise, 1 = data)")
    ax.set_ylabel("||u(x_t, t)||")
    ax.set_title("velocity norm vs t   (spikes at an end = boundary explosion)", fontsize=9)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def save_sampler_matrix(
    grids: dict,
    path: str | Path,
    nrow: int = 4,
) -> None:
    """Solver x budget comparison, one image grid per cell.

    `grids` maps (sampler_name, nfe) -> a batch of images. Columns are
    equal network-call budgets, so cells in a column cost the same
    compute; rows are solvers. Reading across a row shows what more
    compute buys that solver; reading down a column shows which solver
    spends a fixed budget best.
    """
    from torchvision.utils import make_grid

    p = _prep(path)
    samplers = sorted({k[0] for k in grids})
    budgets = sorted({k[1] for k in grids})

    # Match the figure's cell aspect to the image grid's, or tight_layout
    # leaves bands of dead space between rows.
    any_batch = next(iter(grids.values()))
    cell_rows = max(1, (any_batch.shape[0] + nrow - 1) // nrow)
    cell_aspect = cell_rows / nrow  # height / width
    cell_w = 2.6
    fig, axes = plt.subplots(
        len(samplers), len(budgets),
        figsize=(cell_w * len(budgets), cell_w * cell_aspect * len(samplers) + 0.6),
        squeeze=False,
    )
    for r, sname in enumerate(samplers):
        for c, nfe in enumerate(budgets):
            ax = axes[r][c]
            imgs = grids.get((sname, nfe))
            if imgs is not None:
                g = make_grid(((imgs.detach().cpu().clamp(-1, 1) + 1) / 2), nrow=nrow)
                ax.imshow(g.permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{nfe} network calls", fontsize=9)
            if c == 0:
                ax.set_ylabel(sname, fontsize=9)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
