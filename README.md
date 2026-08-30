# Understanding Diffusion & Flow Matching Models

Diffusion and flow matching models, implemented from scratch, to build
real understanding of both formulations -- and a base to try new ideas
from the literature on top of.

## Why this structure

Diffusion (DDPM) and flow matching share almost all their machinery: a
neural net that predicts a vector field, a rule for corrupting data
into noise, a training objective, and a sampler that reverses the
corruption. What differs between them is small and swappable, so the
code is split that way:

```
src/dfm/
  unet.py           the model -- shared, unmodified across processes
  ddpm.py           DDPM: discrete-time schedule, eps-prediction, ancestral + DDIM sampling
  flow_matching.py  rectified flow: continuous-time straight-line path, velocity prediction, ODE sampling
  data.py           Fashion-MNIST loading (scaled to [-1, 1])
  trainer.py         generic training loop -- doesn't know which process it's driving
  utils.py          seeding, device selection, EMA, image-grid saving
scripts/
  train.py          CLI: python scripts/train.py --process ddpm|flow_matching ...
  sample.py         CLI: generate a sample grid from a checkpoint
notebooks/
  01_explore_ddpm_vs_flow_matching.py   VS Code / Jupyter interactive-window "notebook" (# %% cells)
tests/
  test_shapes.py    fast shape/numerics sanity checks, no dataset needed
```

Both `DDPM` and `RectifiedFlow` expose the same two methods:

```python
process.training_loss(model, x1)              # -> scalar loss
process.sample(model, shape, device, ...)      # -> generated batch
```

so `Trainer` (and any future process you add) is agnostic to which one
it's driving. The UNet itself always takes `(x, t)` with `t` a float in
`[0, 1]` -- DDPM normalizes its discrete step index before calling it,
flow matching's `t` is already continuous -- so the exact same model
class serves both.

## Setup

This repo targets a normal `venv`/`pip`/`uv` workflow so it can be
pulled and run unmodified on a cluster. Locally:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .   # editable install of the `dfm` package
pytest -q             # sanity checks, ~seconds, no dataset download
```

On a CUDA cluster, install torch/torchvision from the CUDA-specific
index instead of the default (see the comment at the top of
`requirements.txt`) -- everything else is unchanged, since
`dfm.utils.get_device()` auto-detects CUDA > MPS > CPU.

## Training

```bash
python scripts/train.py --process ddpm --epochs 20
python scripts/train.py --process flow_matching --epochs 20

# quick smoke test (a couple hundred steps on a small subset):
python scripts/train.py --process ddpm --max-steps 200 --subset 512 --epochs 1
```

Checkpoints and periodic sample grids land in `runs/<process>/`.

## Sampling from a checkpoint

```bash
python scripts/sample.py --checkpoint runs/ddpm/checkpoint.pt --process ddpm --sampler ddim --steps 50 --out ddpm_grid.png
python scripts/sample.py --checkpoint runs/flow_matching/checkpoint.pt --process flow_matching --steps 50 --out fm_grid.png
```

## Roadmap: ideas to try next

Roughly in order of how self-contained they are to add, given the
structure above:

1. **DDIM / faster samplers** -- already included for DDPM
   (`DDPM.ddim_sample`); try a proper multi-step solver (DPM-Solver++)
   for flow matching too.
2. **v-prediction** instead of eps-prediction for DDPM -- change the
   training target and the sampler's algebra (`ddpm.py` has a NOTE at
   the spot to change).
3. **Class conditioning + classifier-free guidance** -- add a label
   embedding to `UNet`, randomly drop it during training, blend
   conditional/unconditional predictions at sample time.
4. **EDM-style preconditioning** (Karras et al. 2022) -- reparameterize
   the model's input/output scaling as a function of the noise level;
   unifies a lot of the DDPM-vs-other-schedule design space.
5. **Reflow / distillation for flow matching** -- sample pairs
   `(x0, x1)` from a trained rectified-flow model, retrain on those
   straightened pairs to get few-step (even 1-step) generation.
6. **Consistency models** -- a new process class trained to map any
   point on a trajectory directly to the trajectory's endpoint.
7. **A better UNet** -- more attention blocks, a diffusion-transformer
   variant, or a different backbone entirely -- since the process
   classes don't care about the model's internals, only its `(x, t)`
   interface.

## Notes on this environment

This code was scaffolded and smoke-tested in a small CPU-only sandbox,
which is enough to verify correctness on tiny runs but not enough to
train to convergence. Do real training runs on a GPU (locally if you
have one, or on the cluster once you `git pull` this repo there).
