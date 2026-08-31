"""Axis 3 of 3: the **sampler** -- how the learned field is integrated.

Every sampler here solves the same ODE

    dx/dt = v_theta(x, t),   x(0) ~ N(0, I),   t: 0 -> 1

and returns x(1). Samplers are plain functions of
(model, path, target, ...) rather than methods on a process object, so
you can train once and decode many ways -- comparing solvers on a fixed
checkpoint is one of the cheapest and most informative experiments
available, and it only works if this axis is genuinely independent.

Step count is the knob that exposes the difference: at 100+ steps every
solver here agrees, and the interesting region is 2-20 steps, where
discretisation error dominates and higher-order methods pull ahead.
"""
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from .paths import Path
from .targets import Target


def _velocity_fn(model, path: Path, target: Target) -> Callable:
    """Wrap the network so it always hands back dx/dt, whatever it predicts."""

    def v(x: torch.Tensor, t_scalar: float) -> torch.Tensor:
        t = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=torch.float32)
        return target.to_velocity(path, x, t, model(x, t))

    return v


@torch.no_grad()
def euler(
    model,
    path: Path,
    target: Target,
    shape: tuple[int, ...],
    device: torch.device,
    steps: int = 50,
    progress: bool = True,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """First-order (explicit Euler) integration: x <- x + v * dt.

    One network call per step. Error per step is O(dt^2), so total error
    is O(dt) -- halving the step size roughly halves the error.
    """
    v = _velocity_fn(model, path, target)
    x = torch.randn(shape, device=device)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    it = range(steps)
    if progress:
        it = tqdm(it, total=steps, desc="euler")
    for i in it:
        t, dt = ts[i].item(), (ts[i + 1] - ts[i]).item()
        x = x + v(x, t) * dt
        if return_trajectory:
            traj.append(x.clone())

    return torch.stack(traj) if return_trajectory else x


@torch.no_grad()
def heun(
    model,
    path: Path,
    target: Target,
    shape: tuple[int, ...],
    device: torch.device,
    steps: int = 50,
    progress: bool = True,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Second-order Heun / improved Euler -- the sampler EDM settled on.

    Takes an Euler step to look ahead, then averages the velocity at
    both ends of the interval:

        v1 = v(x, t)
        v2 = v(x + v1*dt, t + dt)
        x <- x + dt * (v1 + v2) / 2

    Two network calls per step, but error O(dt^2) overall. At a fixed
    *compute* budget it usually beats Euler below about 20 steps, which
    is exactly where you want to be. Comparing them at equal step count
    is unfair to Euler; compare at equal network calls.
    """
    v = _velocity_fn(model, path, target)
    x = torch.randn(shape, device=device)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    it = range(steps)
    if progress:
        it = tqdm(it, total=steps, desc="heun")
    for i in it:
        t, t_next = ts[i].item(), ts[i + 1].item()
        dt = t_next - t
        v1 = v(x, t)
        v2 = v(x + v1 * dt, t_next)
        x = x + dt * 0.5 * (v1 + v2)
        if return_trajectory:
            traj.append(x.clone())

    return torch.stack(traj) if return_trajectory else x


SAMPLERS = {"euler": euler, "heun": heun}
