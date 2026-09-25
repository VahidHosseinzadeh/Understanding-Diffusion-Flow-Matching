# Project context for Claude sessions

Flow matching built from scratch for learning, factored along the axes
from Karras et al. 2022 (EDM). Fashion-MNIST and 2D toy distributions.
See README.md for the full picture; this file is conventions for
anyone (human or Claude) editing the code.

## Core design invariant

A generative process here is three *independent* choices. Nothing may
collapse them back into a single object:

```
PATH     paths.py     x_t = alpha(t)*x_data + beta(t)*x_noise
TARGET   targets.py   what the net predicts at (x_t, t)
SAMPLER  samplers.py  how dx/dt = v_theta(x, t) is integrated
```

The test of the split: adding a method must not require editing
`dfm/losses.py`, `trainer.py`, or any existing sampler. If a change forces
one of those, the abstraction is wrong -- fix the abstraction, do not
add a branch.

Contracts:

- `Path` subclasses implement `alpha`, `beta`, `alpha_dot`, `beta_dot`
  and get `interpolate`, `velocity`, `det`, `solve` for free.
- `Target` subclasses implement `regression_target`, `to_velocity` and
  `velocity_error_scale` (the slope of `to_velocity` in the prediction,
  which lets the loss be measured in any target's space), and override
  `t_range` if inverting the interpolant is singular at an endpoint (as
  it is for every target except velocity).
- Samplers are **functions** `(model, path, target, shape, device, ...)`,
  never methods on a process, so any checkpoint can be decoded any way.
- Models implement `forward(x, t) -> same-shape tensor`, with `t` a
  float tensor in `[0, 1]` -- never a raw integer timestep.
- `Trainer` takes `loss_fn(model, batch)` and an optional
  `preview_fn(model, out_dir, epoch)`. It must stay ignorant of both
  the method and the data modality (no hardcoded image shapes).

## Time convention

**t = 0 is noise, t = 1 is data**, everywhere. Sampling integrates
forward, 0 -> 1. The DDPM literature runs the other way; a
variance-preserving path must flip its schedule to match rather than
special-casing samplers.

## Where a new idea goes

- New schedule / corruption -> new `Path` subclass in `dfm/paths.py`.
- New parameterisation -> new `Target` in `dfm/targets.py`. velocity,
  x_data and noise are implemented; score prediction is sketched there.
  A target that inverts the interpolant must declare its `t_range`.
- New solver (DDIM, DPM-Solver, RK4) -> new function in `dfm/samplers.py`,
  registered in `SAMPLERS`, with its cost in `NFE_PER_STEP` (and
  `NFE_LAST_STEP` if its final step differs) so budget comparisons stay
  fair. Convert between steps and calls only via `network_calls` /
  `steps_for_budget`, never `steps * 2`.
- New diagnostic -> `dfm/diagnostics.py` returns plain tensors, `dfm/viz.py`
  renders them, `train.py` wires them behind `--diagnostics`. Keep the
  three separable; the trainer must not learn what is being measured.
- New t-distribution or loss weighting -> `dfm/losses.py`. A weighting
  is `w(path, target, t) -> (B,)`. Measuring the loss in another
  target's space is already one (`LossSpaceWeighting`, `--loss-space`),
  and a new Target becomes a loss space with no change there.
- New sample-quality metric -> `dfm/metrics.py` (uint8 images in, a
  dict of floats out), wired in `evaluate.py`. torch-fidelity stays
  optional like wandb: imported lazily, and no test runs Inception.
  Never give generated samples a torch-fidelity cache name -- every
  later checkpoint would silently reuse the first one's features.
- Loading a checkpoint -> `checkpoint.load_checkpoint`, the one place
  that rebuilds model, path and target from `meta`.
- New architecture -> new file, must satisfy `forward(x, t)`.
- Conditioning -> extend the model's `forward`, thread the condition
  through `interpolant_loss` and the samplers; keep the unconditional
  path working via a `None` default.

## Environment

- Flat layout: everything in `dfm/`, imported by plain name
  (`from paths import LinearPath`). No `__init__.py`, no package, no
  install -- `train.py` sits beside the modules so Python finds them.
  Do not add `__init__.py`; it would break every import.
- Same setup on macOS and the cluster; see README. No platform-specific
  torch index needed.
- Tests: `pytest -q` -- ~1s, no dataset download. These check numerics
  (finite-difference derivatives, solver exactness on closed-form
  fields), not just shapes. Run after any change to `dfm/paths.py`,
  `dfm/targets.py`, `dfm/samplers.py`, or `dfm/losses.py`.
- Start experiments on 2D (`--data moons`), where the velocity field is
  directly plottable. Move to `--data fashion_mnist` after.
- Real training and evaluation belong on a GPU machine;
  `--max-steps N --subset M` is the smoke-test pattern here.
  torch-fidelity has no MPS path, so FID on a Mac runs on the CPU.
- Tracking (`dfm/tracking.py`) is optional and must stay that way:
  `wandb` is imported lazily, `NullTracker` is the default, and local
  PNGs/`losses.json` are written regardless. Never make a code path
  depend on a tracker being present.

## Style

- Small, readable modules over cleverness -- this repo is for building
  understanding, not for being terse.
- Type hints on public functions/methods.
- Docstrings should say *why*, and name the paper where one applies.
- Every path/target/sampler addition gets a numerical test, not just a
  shape test.
