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

from paths import Path, expand_to, floor_magnitude
from targets import Target


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
    t0, t1 = target.t_range()
    ts = torch.linspace(t0, t1, steps + 1, device=device)
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

    -- except on the last step, which is plain Euler. EDM ends every
    trajectory the same way (Karras et al. 2022, Algorithm 1: no
    correction on the step onto sigma = 0), for the same reason: the
    correction would evaluate the network at the data endpoint, where the
    learned field is least reliable (on the linear path the velocity is
    (x_hat - x_t) / (1 - t)). Averaging that in left visible noise in
    Fashion-MNIST samples -- FID 36.7 against Euler's 10.2 at 20 network
    calls. One first-order step still leaves the global error O(dt^2).

    So N steps cost 2N - 1 network calls (`network_calls`). At a fixed
    *compute* budget it usually beats Euler below about 20 steps, which
    is exactly where you want to be. Comparing them at equal step count
    is unfair to Euler; compare at equal network calls.
    """
    v = _velocity_fn(model, path, target)
    x = torch.randn(shape, device=device)
    t0, t1 = target.t_range()
    ts = torch.linspace(t0, t1, steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    it = range(steps)
    if progress:
        it = tqdm(it, total=steps, desc="heun")
    for i in it:
        t, t_next = ts[i].item(), ts[i + 1].item()
        dt = t_next - t
        v1 = v(x, t)
        if i == steps - 1:
            x = x + dt * v1  # never evaluate the network at the endpoint
        else:
            v2 = v(x + v1 * dt, t_next)
            x = x + dt * 0.5 * (v1 + v2)
        if return_trajectory:
            traj.append(x.clone())

    return torch.stack(traj) if return_trajectory else x


@torch.no_grad()
def euler_maruyama(
    model,
    path: Path,
    target: Target,
    shape: tuple[int, ...],
    device: torch.device,
    steps: int = 50,
    progress: bool = True,
    return_trajectory: bool = False,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Stochastic sampling -- the "SDE extension" of the same model.

    Every ODE above is the probability-flow form of a whole family of
    SDEs that share its marginals (Song et al. 2021). Adding noise back
    in, and correcting the drift with the score to compensate:

        dX = [u(X,t) + (sigma^2 / 2) * score(X,t)] dt + sigma dW

    `sigma` here is the SDE diffusion coefficient -- the sigma_t that
    `paths.py` deliberately reserves the name for. It is a *sampling*
    knob on an already-trained model: sigma=0 recovers `euler` exactly,
    and larger values trade determinism for extra stochastic correction,
    which can clean up a model whose learned field is slightly off.

    The score comes free from the velocity. Inverting the interpolant
    gives the noise endpoint, and for a Gaussian path

        score = -x_noise / beta(t)

    so this works with any target, not just the ones that predict noise
    directly. Note the division: score blows up as beta(t) -> 0, hence
    `floor_magnitude`. One network call per step, same as `euler`.
    """
    v = _velocity_fn(model, path, target)
    x = torch.randn(shape, device=device)
    t0, t1 = target.t_range()
    ts = torch.linspace(t0, t1, steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    it = range(steps)
    if progress:
        it = tqdm(it, total=steps, desc="euler-maruyama")
    for i in it:
        t_scalar, dt = ts[i].item(), (ts[i + 1] - ts[i]).item()
        t_vec = torch.full((x.shape[0],), t_scalar, device=device)

        velocity = v(x, t_scalar)
        _, x_noise = path.solve(x, velocity, t_vec)
        beta_t = floor_magnitude(expand_to(path.beta(t_vec), x))
        score = -x_noise / beta_t

        drift = velocity + 0.5 * sigma**2 * score
        x = x + drift * dt + sigma * (dt ** 0.5) * torch.randn_like(x)
        if return_trajectory:
            traj.append(x.clone())

    return torch.stack(traj) if return_trajectory else x


SAMPLERS = {"euler": euler, "heun": heun, "euler_maruyama": euler_maruyama}

# Network calls per step, so budgets can be compared fairly. Heun evaluates
# the field twice per step; comparing solvers at equal *steps* silently
# hands it double the compute.
NFE_PER_STEP = {"euler": 1, "heun": 2, "euler_maruyama": 1}

# Where a solver's final step costs something else. Heun's is plain Euler
# (see `heun`), so N heun steps cost 2N - 1 calls -- the count EDM reports.
NFE_LAST_STEP = {"heun": 1}


def network_calls(name: str, steps: int) -> int:
    """Exactly how many times SAMPLERS[name] evaluates the network."""
    per_step = NFE_PER_STEP[name]
    return per_step * (steps - 1) + NFE_LAST_STEP.get(name, per_step)


def steps_for_budget(name: str, budget: int) -> int:
    """The most steps SAMPLERS[name] can take within `budget` network calls."""
    per_step = NFE_PER_STEP[name]
    return max(1, (budget - NFE_LAST_STEP.get(name, per_step)) // per_step + 1)
