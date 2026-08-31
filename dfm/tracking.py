"""Experiment tracking, behind a two-method interface.

The package must run with no tracker and no `wandb` installed -- local
PNGs and `losses.json` stay the source of truth, and a tracker only
*mirrors* them. That ordering matters for a learning repo: your
artifacts survive losing interest in the tool, changing accounts, or
working on a machine with no network.

What a tracker buys you that local files do not:

  - runs side by side. Train the same data under different paths or
    targets and overlay the loss curves, rather than opening PNGs in
    two windows.
  - a config table. Every axis is logged to `wandb.config`, so the run
    table can be grouped or filtered by `path`, `target`, `t_dist`,
    which is how you answer "did changing the path actually help?"
  - remote runs. Training on the cluster and watching from a browser,
    instead of scp-ing image files back.

Adding a backend means one more subclass; nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    """Minimal surface: scalars, images, teardown."""

    @abstractmethod
    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...

    @abstractmethod
    def log_images(self, step: int, images: dict[str, str | Path]) -> None:
        """`images` maps a panel name to an already-saved image file."""

    def finish(self) -> None:
        """Flush and close. Safe to call more than once."""


class NullTracker(Tracker):
    """The default. Does nothing, so the trainer needs no branches."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        pass

    def log_images(self, step: int, images: dict[str, str | Path]) -> None:
        pass


class WandbTracker(Tracker):
    """Weights & Biases.

    `wandb` is imported here, not at module import, so the package works
    when it is not installed.

    AUTH -- never put a key in the code or in a commit:
        local:    wandb login          (writes ~/.netrc)
        cluster:  export WANDB_API_KEY=...   (from your shell profile
                  or the scheduler's secret store)
        offline:  export WANDB_MODE=offline  (logs to disk; upload later
                  with `wandb sync`). Useful on compute nodes with no
                  outbound network.
    """

    def __init__(
        self,
        project: str = "dfm",
        name: str | None = None,
        group: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "wandb is not installed. Run: pip install wandb"
            ) from exc

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,      # ties a set of variations together
            entity=entity,
            tags=tags,
            config=config or {},
            dir=str(out_dir) if out_dir else None,
        )
        self._finished = False

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self._wandb.log(values, step=step)

    def log_images(self, step: int, images: dict[str, str | Path]) -> None:
        self._wandb.log(
            {k: self._wandb.Image(str(v)) for k, v in images.items()}, step=step
        )

    def finish(self) -> None:
        if not self._finished:
            self._wandb.finish()
            self._finished = True


def make_tracker(
    backend: str = "none",
    **kwargs: Any,
) -> Tracker:
    """Factory used by the CLI. `backend='none'` gives a NullTracker."""
    if backend in ("none", None):
        return NullTracker()
    if backend == "wandb":
        return WandbTracker(**kwargs)
    raise ValueError(f"unknown tracking backend: {backend!r}")


TRACKERS = ["none", "wandb"]
