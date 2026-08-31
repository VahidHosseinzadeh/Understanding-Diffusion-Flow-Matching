# Flow Matching from Scratch

Flow matching implemented from scratch to build real understanding, and
structured so that diffusion drops in later as a variation rather than
a rewrite.

## The three axes

Following Karras et al. 2022, *[Elucidating the Design Space of
Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)*
(EDM): what looks like a zoo of separate methods -- DDPM, DDIM,
rectified flow, score SDEs -- is a small number of independent choices.
This package keeps them independent:

| axis | file | question it answers |
|---|---|---|
| **path** | `paths.py` | how do noise and data interpolate? |
| **target** | `targets.py` | what does the network predict? |
| **sampler** | `samplers.py` | how is the learned field integrated? |

```
x_t = alpha(t) * x_data + sigma(t) * x_noise        <- path
      network predicts velocity / x_data / x_noise  <- target
      dx/dt = v_theta(x, t),  t: 0 -> 1             <- sampler
```

Independence is the whole point. Samplers are plain functions, not
methods on a process object, so one checkpoint can be decoded many
ways. The loss is written against `(path, target)`, so it never changes
when you add a method. `Trainer` knows only `loss_fn(model, batch)`.

**Currently implemented:** `LinearPath` + `VelocityTarget` = rectified
flow, with Euler and Heun samplers. The other slots are empty on
purpose, with the derivations written into the docstring where each one
goes -- `targets.py` shows the four lines that make eps-prediction work.

**Time convention:** `t = 0` is noise, `t = 1` is data, everywhere.
(DDPM literature runs the other way. A VP path must flip its schedule
to match, rather than special-casing the samplers.)

```
src/dfm/
  paths.py      alpha(t), sigma(t) and derivatives; interpolate/velocity/solve
  targets.py    what the net regresses onto, and how to get dx/dt back
  samplers.py   euler, heun
  losses.py     the MSE objective, t-distribution, per-timestep weighting
  embeddings.py sinusoidal time conditioning, shared by both models
  mlp.py        model for 2D toy data
  unet.py       model for images
  data.py       moons / eight_gaussians / spiral, and Fashion-MNIST
  trainer.py    training loop; knows nothing about flow matching
  viz.py        velocity fields, trajectories, scatters, loss curves
  utils.py      seeding, device, EMA
scripts/
  train.py      python scripts/train.py --data moons
  sample.py     python scripts/sample.py --checkpoint ... --sampler heun
tests/          numerical tests, not just shape tests
```

## Setup

The same three commands work on macOS and on the GPU cluster -- only
stdlib `venv` + `pip` are required, and `requirements.txt` needs no
platform-specific index (PyPI serves an MPS-capable torch wheel on
macOS arm64 and a CUDA-bundled one on Linux x86_64):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .      # editable install of the `dfm` package
pytest -q             # sanity checks, ~1s, no dataset download
```

`.venv/` is gitignored and is **not** portable between machines -- a
venv hardcodes absolute paths to its base interpreter, so a venv built
on the cluster will not run on the Mac (and vice versa). Create one per
machine; only the repo travels.

Everything else is unchanged across machines because
`dfm.utils.get_device()` auto-detects CUDA > MPS > CPU at runtime.
Verify with:

```bash
python -c "from dfm.utils import get_device; print(get_device())"
```

Python 3.10+ is required (`pyproject.toml`); this repo is currently
developed on 3.11. If the cluster pins an older CUDA than the bundled
wheel expects, override just torch with the matching index -- see the
comment at the top of `requirements.txt`.

## Training

**Start in 2D.** You can plot the entire learned velocity field, and it
trains in under a minute on CPU:

```bash
python scripts/train.py --data moons --epochs 80 --device cpu
```

This writes, into `runs/moons_linear_velocity/`:

- `field_epoch*.png` -- the learned velocity field at t = 0, .25, .5, .75, 1.
  **This is the plot to look at.** Early in t the field should sweep
  broadly inward from everywhere; by t = 1 it should be near zero on the
  data manifold and still pointing inward off it.
- `samples_epoch*.png` -- generated points over the true distribution
- `loss_curve.png`, `losses.json` -- per-step loss, raw and smoothed

Other toys: `--data eight_gaussians` (mode coverage is obvious),
`--data spiral` (high curvature; hardest for straight-line paths).

Then images:

```bash
python scripts/train.py --data fashion_mnist --model unet --epochs 20
```

Smoke test in seconds: `--max-steps 50 --epochs 1 --subset 512`.

## Sampling

Path, target and model are read back from the checkpoint, so the only
thing you choose is how to decode:

```bash
python scripts/sample.py --checkpoint runs/moons_linear_velocity/checkpoint.pt \
    --sampler euler --steps 50 --trajectories
```

`--trajectories` plots the noise-to-data paths. For rectified flow they
should be close to straight -- that is the property the method is named
for, and seeing how straight they actually are is the point.

Compare solvers at equal *network calls*, not equal steps (Heun uses
two per step):

```bash
python scripts/sample.py --checkpoint ... --sampler euler --steps 10   # 10 calls
python scripts/sample.py --checkpoint ... --sampler heun  --steps 5    # 10 calls
```

At 100+ steps every solver agrees. The interesting region is 2-20.

## Experiment tracking (optional)

Everything above works with no tracker: local PNGs and `losses.json`
are always written. A tracker only *mirrors* them, so your artifacts
never depend on the tool.

```bash
pip install -e ".[tracking]"
wandb login          # stores the key in ~/.netrc -- never commit a key
```

```bash
python scripts/train.py --data moons --epochs 80 --tracker wandb
```

Loss goes up every `--log-every-steps`; the sample scatter and the
velocity field are logged as image panels each preview, so you can
scrub them across training in the browser instead of opening PNGs.

**Comparing variations** is the reason to bother. Every axis is written
to `wandb.config` (`path`, `target`, `sigma_min`, `t_dist`, `model`,
`seed`, ...), so the run table can be grouped or filtered by it. Use
`--wandb-group` to tie a sweep together:

```bash
for p in linear vp; do
  python scripts/train.py --data fashion_mnist --path $p \
      --tracker wandb --wandb-group path-sweep
done
```

Both runs log to the same panel names, so `samples` and
`velocity_field` line up side by side, and the loss curves overlay.

On a compute node with no outbound network:

```bash
export WANDB_MODE=offline    # logs to <out_dir>/wandb/
wandb sync <out_dir>/wandb/offline-run-*    # upload later
```

On the cluster, set `WANDB_API_KEY` from your shell profile or the
scheduler's secret store rather than running `wandb login`.

## Tests

```bash
pytest -q     # ~1s, no dataset download
```

These check numerics, not just shapes -- `velocity()` is verified
against a finite-difference derivative of `interpolate()`, and the
samplers are checked against velocity fields with closed-form
solutions. A path whose `alpha_dot` disagrees with its `alpha` would
otherwise train a subtly wrong field and still make plausible pictures.

## Suggested path through it

1. Read `paths.py`, then `losses.py`. That is the whole method: draw
   noise, draw a time, interpolate, regress.
2. Train on `moons` and watch `field_epoch*.png` across epochs.
3. Implement `DataTarget` and `NoiseTarget` in `targets.py` (four lines
   each -- `Path.solve` does the work). Train all three and compare.
   Note where they blow up at the endpoints; that is what EDM's
   preconditioning exists to fix.
4. Add a variance-preserving `Path` -- that is DDPM, and it should
   require no change to the loss, the trainer, or any sampler.
