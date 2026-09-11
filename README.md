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
x_t = alpha(t) * x_data + beta(t) * x_noise         <- path
      network predicts velocity / x_data / x_noise  <- target
      dx/dt = v_theta(x, t),  t: 0 -> 1             <- sampler
```

Independence is the whole point. Samplers are plain functions, not
methods on a process object, so one checkpoint can be decoded many
ways. The loss is written against `(path, target)`, so it never changes
when you add a method. `Trainer` knows only `loss_fn(model, batch)`.

**Currently implemented:** `LinearPath`, three targets (`velocity`,
`x_data`, `noise`), and Euler + Heun samplers. So the target axis is
live: same path, same data, same sampler, three parameterisations of
the same model, selected with `--target`. Score prediction is sketched
in `targets.py` where it goes.

**Time convention:** `t = 0` is noise, `t = 1` is data, everywhere.
(DDPM literature runs the other way. A VP path must flip its schedule
to match, rather than special-casing the samplers.)

```
dfm/
  paths.py      alpha(t), beta(t) and derivatives; interpolate/velocity/solve
  targets.py    what the net regresses onto, and how to get dx/dt back
  samplers.py   euler, heun, euler_maruyama (SDE)
  diagnostics.py loss-vs-t, straightness, velocity norms, budget matrix
  losses.py     the MSE objective, t-distribution, per-timestep weighting
  embeddings.py sinusoidal time conditioning, shared by both models
  mlp.py        model for 2D toy data
  unet.py       model for images
  dataset.py    moons / eight_gaussians / spiral, and Fashion-MNIST
  trainer.py    training loop; knows nothing about flow matching
  viz.py        velocity fields, trajectories, scatters, loss curves
  tracking.py   optional Weights & Biases logging
  utils.py      seeding, device, EMA
  train.py      python dfm/train.py --data moons
  sample.py     python dfm/sample.py --checkpoint ... --sampler heun
  tests/        numerical tests, not just shape tests
train.sbatch    Slurm job for the image runs
data/           Fashion-MNIST lands here (gitignored)
runs/           checkpoints, previews, loss curves (gitignored)
```

Flat on purpose: everything lives in `dfm/` and imports its siblings by
plain name (`from paths import LinearPath`). `train.py` sits in there
too, and Python puts a script's own folder on the path -- so
`python dfm/train.py` just runs. No package, no `__init__.py`, no
`pip install`.

## Setup

Nothing to install. You need Python with torch, matplotlib and tqdm --
on a cluster that is whatever environment already has them:

```bash
module load python/3.11      # if your cluster needs it
source activate torch2.2     # or whatever your env is called
pytest -q                    # 26 tests, ~2s
```

Locally, if you have no torch yet:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

`.venv/` is gitignored and is not portable between machines -- create
one per machine, never copy it to the cluster. Only the repo travels.

`utils.get_device()` picks CUDA > MPS > CPU, so no code changes between
machines:

```bash
PYTHONPATH=dfm python -c "from utils import get_device; print(get_device())"
```

## Training

**Start in 2D.** You can plot the entire learned velocity field, and it
trains in under a minute on CPU:

```bash
python dfm/train.py --data moons --epochs 80 --device cpu
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
python dfm/train.py --data fashion_mnist --model unet --epochs 20
```

Smoke test in seconds: `--max-steps 50 --epochs 1 --subset 512`.

## Sampling

Path, target and model are read back from the checkpoint, so the only
thing you choose is how to decode:

```bash
python dfm/sample.py --checkpoint runs/moons_linear_velocity/checkpoint.pt \
    --sampler euler --steps 50 --trajectories
```

`--trajectories` plots the noise-to-data paths. For rectified flow they
should be close to straight -- that is the property the method is named
for, and seeing how straight they actually are is the point.

Compare solvers at equal *network calls*, not equal steps (Heun uses
two per step):

```bash
python dfm/sample.py --checkpoint ... --sampler euler --steps 10   # 10 calls
python dfm/sample.py --checkpoint ... --sampler heun  --steps 5    # 10 calls
```

At 100+ steps every solver agrees. The interesting region is 2-20.

## Experiment tracking (optional)

Everything above works with no tracker: local PNGs and `losses.json`
are always written. A tracker only *mirrors* them, so your artifacts
never depend on the tool.

```bash
pip install wandb
wandb login          # stores the key in ~/.netrc -- never commit a key
```

```bash
python dfm/train.py --data moons --epochs 80 --tracker wandb
```

Loss goes up every `--log-every-steps`; the sample scatter and the
velocity field are logged as image panels each preview, so you can
scrub them across training in the browser instead of opening PNGs.

**Comparing variations** is the reason to bother. Every axis is written
to `wandb.config` (`path`, `target`, `beta_min`, `t_dist`, `model`,
`seed`, ...), so the run table can be grouped or filtered by it. Use
`--wandb-group` to tie a sweep together:

```bash
for p in linear vp; do
  python dfm/train.py --data fashion_mnist --path $p \
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

### Targets are not interchangeable in practice

Mathematically the three targets are one model. Empirically they are
not, and the reason is worth understanding before you trust any of
them.

`to_velocity` has to invert the interpolant for every parameterisation
except velocity, and that division blows up where its coefficient
vanishes -- `alpha(0) = 0` for noise-prediction, `beta(1) = 0` for
x_data-prediction. Both are 0/0, so the answer is finite in exact
arithmetic, but any error the network has gets amplified without bound
as you approach the bad end.

Measured on 2D moons, 60 epochs, identical seed, mean distance from a
sample to the nearest real data point (lower is better):

| target | euler | heun |
|---|---|---|
| `velocity` | 0.046 | 0.045 |
| `x_data` | 0.104 | 0.106 |
| `noise` | 0.491 | 0.473 |

Before each target declared a `t_range` to keep samplers off its
singular endpoint, the same numbers were **3039** for `noise` + euler
(which evaluates at t=0) and **5.3** for `x_data` + heun (whose
look-ahead lands on t=1). Silently finite, entirely wrong -- the
magnitude floor in `targets.py` turns a NaN into plausible-looking
garbage, which is worse than a crash.

Even with the endpoints trimmed, noise-prediction stays ~10x behind
velocity here. That is not a bug to fix: it is why flow matching
prefers velocity, why DDPM's own samplers work directly in eps space
instead of converting, and what EDM's preconditioning is for.

## Diagnostics

Sample grids tell you whether a model works. They do not tell you
*where* it is failing, and above 2D there is no velocity-field plot to
fall back on. `--diagnostics` adds four probes, logged locally as PNGs
and mirrored to wandb:

```bash
python dfm/train.py --data fashion_mnist --diagnostics --tracker wandb
```

| panel | what it answers |
|---|---|
| `loss_vs_time` | is training uniformly hard across t, or is one band of noise levels soaking up the gradient? 20 bins, fed from the training loss itself -- no extra forward passes |
| `straightness` | how straight are the sampled paths? `\|x1-x0\| / path length`, mean + histogram |
| `velocity_norm_vs_time` | does the drift explode near an endpoint? log-scaled, sweeps the full [0,1] |
| `sampler_matrix` | solver x budget grid at **equal network calls** (images only) |

Plus scalars `straightness_index`, `velocity_norm_mean`, `velocity_norm_max`.

Costs about 5s per preview, so it is off by default. Raise
`--preview-every-epochs` if you want it on a long run.

### What they catch

Trained on 2D moons, 40 epochs, one run per target:

| target | ‖u‖ @t=0 | @t=0.5 | @t=1 | straightness |
|---|---|---|---|---|
| `velocity` | 0.63 | 0.47 | 0.96 | 0.703 |
| `x_data` | 1.10 | 0.53 | **4177** | 0.938 |
| `noise` | **4548** | 0.89 | 1.20 | 0.985 |

Each target explodes at exactly the endpoint its `to_velocity` divides
by zero at -- `beta(1)=0` for x_data, `alpha(0)=0` for noise -- and
velocity, which inverts nothing, stays flat. That is the diagnostic
doing its job.

**Read straightness with care.** Note that `noise` scores *highest*
(0.985) while producing by far the worst samples. Straightness measures
how efficiently a path travels, not whether it arrives anywhere useful:
a model that shoots off in a confident straight line to the wrong place
scores near 1.0. It is evidence about path geometry, which is what
rectified flow claims to improve -- never a substitute for looking at
what came out.

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
3. Compare the three targets: `--target velocity`, `--target x_data`,
   `--target noise` on the same data. They are the same model
   reparameterised, but not equally well behaved -- see "Targets are
   not interchangeable in practice" below.
4. Add a variance-preserving `Path` -- that is DDPM, and it should
   require no change to the loss, the trainer, or any sampler.
