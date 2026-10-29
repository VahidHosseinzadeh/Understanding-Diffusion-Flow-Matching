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
  diagnostics.py loss-vs-t, velocity norms, solver budget matrix
  losses.py     the MSE objective, t-distribution, per-timestep weighting,
                loss space
  embeddings.py time and label embeddings, shared by every model
  mlp.py        model for 2D toy data
  unet.py       model for images
  jit.py        Transformer on pixel patches for images (JiT)
  dataset.py    moons / eight_gaussians / spiral, and Fashion-MNIST
  trainer.py    training loop; knows nothing about flow matching
  viz.py        velocity fields, trajectories, scatters, loss curves
  tracking.py   optional Weights & Biases logging
  utils.py      seeding, device, EMA
  checkpoint.py rebuilds model, path and target from a checkpoint's meta
  convert.py    foreign weights -> a checkpoint of ours (Li & He's JiT)
  metrics.py    FID / KID / IS / precision-recall, via torch-fidelity
  train.py      python dfm/train.py --data moons
  sample.py     python dfm/sample.py --checkpoint ... --sampler heun
  evaluate.py   python dfm/evaluate.py --checkpoint ... --nfe 10 20 50
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
pytest -q                    # ~2s
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

- `filmstrip_epoch*.png` -- **noise → data in a row.** One row per
  sample, time running left to right, so you see a single generation
  unfold rather than a finished grid. Comes free: the preview already
  samples, and the filmstrip is that same trajectory. `--filmstrip-frames 0`
  turns it off, any other number sets the column count.
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
two per step, one on its last):

```bash
python dfm/sample.py --checkpoint ... --sampler euler --steps 9    # 9 calls
python dfm/sample.py --checkpoint ... --sampler heun  --steps 5    # 9 calls
```

At 100+ steps every solver agrees. The interesting region is 2-20.

## Class conditioning and classifier-free guidance

Train with the label, dropped with probability 0.1 (`--label-dropout`),
so one network learns both the class-conditional fields and the
unconditional one:

```bash
python dfm/train.py --data fashion_mnist --class-cond
CLASS_COND=1 sbatch train.sbatch            # on the cluster
```

Then sample with guidance weight w (Ho & Salimans 2022):

```bash
python dfm/sample.py --checkpoint runs/<run>/checkpoint.pt --cfg-scale 3
python dfm/sample.py --checkpoint ... --cfg-scale 3 --label 7      # sneakers only
python dfm/sample.py --checkpoint ... --cfg-scale 3 --guidance-interval 0.2 1
```

The guided prediction is `(1 - w) * uncond + w * cond`: w = 1 is plain
conditional sampling, w = 0 is the same network's unconditional field,
and w > 1 pushes away from the unconditional prediction. The grid has
one class per column. `--guidance-interval LO HI` guides only while
LO <= t <= HI (Kynkäänniemi et al. 2024; t = 0 is noise here) and is
plainly conditional elsewhere.

This works with every target and every sampler unchanged. The
conditioning goes into the network, not the method: the label
embedding is added to the time embedding, so a denoiser learns
E[x_data | x_t, y] and a noise predictor E[noise | x_t, y]. Guidance
mixes the network's two outputs before any conversion to velocity.
That is exact because every target converts to velocity affinely and
the two weights sum to one (checked for every target in
`test_guidance.py`), so `guidance.ClassifierFreeGuidance` simply wraps
the network.

Guidance doubles the work per step: the conditional and unconditional
predictions come from one call on a doubled batch. Each run prints the
`network passes` it spent beside the `network calls`, and an interval
saves the passes outside it.

## A Transformer instead of the UNet (JiT)

`--model jit` replaces the UNet with JiT (Li & He 2025, "Back to Basics:
Let Denoising Generative Models Denoise"). It cuts the image into p x p
patches, makes each patch one token, runs a plain Transformer whose
blocks are conditioned by adaLN-Zero (DiT, Peebles & Xie 2023), and
projects each token back to its patch. Path, target, sampler and
guidance are untouched: the model keeps `forward(x, t, y=None)`.

```bash
# JiT's recipe: x-prediction, velocity loss, logit-normal t, label tokens
python dfm/train.py --data fashion_mnist --model jit --class-cond --in-context-tokens 8 \
    --target x_data --loss-space velocity --t-dist logit_normal --t-mean -0.8 --t-std 0.8
python dfm/sample.py --checkpoint runs/<run>/checkpoint.pt --sampler heun --steps 50 \
    --cfg-scale 3 --guidance-interval 0.1 1
```

The defaults are 4x4 patches (49 tokens of 16 numbers), width 256
(`--hidden`), 6 blocks (`--depth`) and 4 heads: 7.4M parameters, against
the UNet's 6.1M. `--t-mean -0.8` is JiT's value for 256x256 ImageNet,
where more noise helped; 28x28 images may want less, and the default is 0.
The training loop is this repo's (AdamW, EMA 0.999, no warmup), at
DiT's learning rate for batch 256, 1e-4.

JiT predicts x_data because a 16x16x3 ImageNet patch is 768 numbers, as
many as JiT-B has per token. Noise and velocity both contain the noise
sample, so predicting them means carrying all 768 numbers through every
block. A clean patch needs far fewer. At 4x4 a token is 16 numbers
against 256 channels, which is the regime where JiT found every target
works. To bring the bottleneck back, make the patches wider than the
tokens, then compare `--target x_data --loss-space velocity` against
`--target velocity`:

```bash
python dfm/train.py --data fashion_mnist --model jit --patch-size 14 --hidden 128 ...
```

This gives 4 tokens of 196 numbers at width 128. `--bottleneck-dim 32` instead
embeds each patch through 32 numbers on purpose, which JiT found
harmless for x-prediction.

### Li & He's pretrained weights

`jit.JIT_PRESETS` rebuilds the paper's six ImageNet models exactly. B, L
and H are ViT's Base, Large and Huge sizes; /16 and /32 are the patch
size. `convert.py` turns their released weights into an ordinary checkpoint
of ours. After that, sample.py and every sampler use them unchanged:

```bash
# official release (one folder per model on the authors' Dropbox; B/16 is 1.5 GB:
# raw weights plus two EMAs), or the single-weight mirror BiliSakura/JiT-diffusers
python dfm/convert.py jit-official checkpoint-last.pth --preset JiT-B/16 --out jit_b16.pt
python dfm/convert.py jit-diffusers diffusion_pytorch_model.safetensors \
    --preset JiT-B/16 --out jit_b16.pt                   # needs: pip install safetensors

# their sampler settings for B/16: Heun 50 steps, guidance 3 for t in [0.1, 1]
python dfm/sample.py --checkpoint jit_b16.pt --sampler heun --steps 50 \
    --cfg-scale 3 --guidance-interval 0.1 1 --label 207 360 387 974 88 979 417 279 --n 32 \
    --tracker wandb --wandb-group pretrained-jit
```

Two settings differ from our small default, and the presets set both:

- `time_scale=1`: their t enters the time embedding unscaled.
- `norm_gain=True`: their RMSNorms keep a learned gain before adaLN.

The converter checks names, shapes and the fixed position codes rather
than trusting them. `tests/test_convert.py` checks the layouts against
the operations the reference code runs.

Weights from another JiT implementation need one more list of regex
renames in `convert.RENAMES` and a reader. The /32 models are not
supported yet: they were trained with noise of standard deviation 2,
which no Path here provides.

## Evaluation

FID and friends for image checkpoints, computed with
[torch-fidelity](https://github.com/toshas/torch-fidelity) as in JiT --
the standard TF-compatible Inception-v3 protocol, not a lookalike whose
numbers compare with nothing:

```bash
pip install torch-fidelity
python dfm/evaluate.py --checkpoint runs/<run>/checkpoint.pt
python dfm/evaluate.py --checkpoint ... --sampler euler heun --nfe 10 20 50
```

Each (sampler, NFE) cell scores `--n` samples (default 10k) against the
real training split and prints FID, KID and Inception Score. For a
class-conditional checkpoint, every class gets the same number of samples,
and `--cfg-scale 0 1 2 3` sweeps the guidance weight, one cell each. Add `prc`
to `--metrics` for precision/recall. Each cell writes its own
`eval_<reference>_n<n>_<sampler>_nfe<nfe>.json` beside the checkpoint
(suffixed `_raw` or `_s<seed>` for non-default weights or seed),
carrying every setting that affects the numbers, plus a `.png` grid of
the samples actually scored. As in the `sampler_matrix` diagnostic,
budgets count network calls and every cell starts from the same noise.

| metric | what it measures | caveat |
|---|---|---|
| FID | distance between Gaussians fitted to Inception features; lower is better | biased by sample count: compare only at equal `--n` (papers use 50k) |
| KID | the same features, compared by kernel MMD; lower is better | unbiased, so steadier than FID at a few thousand samples |
| IS | how confidently and variedly Inception labels the samples | ImageNet's classes do not describe Fashion-MNIST: relative signal only |
| precision / recall | fidelity and coverage, separately | the two failures FID folds into one number; costs a VGG-16 pass |

Run it on a GPU: on an A100 a 10k-sample cell takes about 40 s, while
on a laptop CPU (torch-fidelity has no MPS path) Inception needed more
than 15 minutes for 12k images. The first run per split also scores
every real image -- about 30 s for the 60k training set on the A100 --
and caches the result in `data/fidelity_cache/` for reuse. Compute nodes without internet need
the Inception weights fetched once from a login node:

```bash
python -c "import torch, torch_fidelity.feature_extractor_inceptionv3 as m; torch.hub.load_state_dict_from_url(m.URL_INCEPTION_V3)"
```

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

The heun column predates heun's last step becoming plain Euler (see
`samplers.heun`): its correction onto t=1 queried the network where the
learned field is least reliable, which on Fashion-MNIST cost FID 36.7
against Euler's 10.2 at 20 network calls.

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

### Where the loss is measured is a separate choice

`--target` fixes what the network outputs. By default the error is
measured in that same space, but it need not be: at a fixed `x_t`,
every target's error is the velocity error times a scalar,

    v_theta - v = s(t) * (pred - target)

with `s = 1` for velocity, `-det/beta` for x_data and `det/alpha` for
noise (`det = alpha*beta' - alpha'*beta`, which is -1 on the linear
path). So measuring a prediction's error in another target's space is
only a per-t weight, `(s_pred / s_space)^2`. The prediction is never
converted, and the loss code does not change:

```bash
# x-prediction with a v-loss, the pairing JiT (Li & He 2025) trains
python dfm/train.py --data fashion_mnist --target x_data --loss-space velocity
```

On the linear path that weight is `1/(1-t)^2`, which diverges at t=1;
`--max-loss-weight` (default 400) caps it. JiT's own code converts both
sides to velocity, dividing by `max(1-t, 0.05)`. That is the same loss:
`x_t` cancels in the difference, and a test transcribes their code to
check it. Sampling is untouched, since the network still predicts x_data.

This also makes loss curves comparable across targets: measure all
three in velocity space and their curves share units (up to the cap).

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
| `velocity_norm_vs_time` | does the drift explode near an endpoint? log-scaled, sweeps the full [0,1] |
| `sampler_matrix` | solver x budget grid at **equal network calls** (images only) |

Plus scalars `velocity_norm_mean` and `velocity_norm_max`.

Costs about 5s per preview, so it is off by default. Raise
`--preview-every-epochs` if you want it on a long run.

### What they catch

Trained on 2D moons, 40 epochs, one run per target:

| target | ‖u‖ @t=0 | @t=0.5 | @t=1 |
|---|---|---|---|
| `velocity` | 0.63 | 0.47 | 0.96 |
| `x_data` | 1.10 | 0.53 | **4177** |
| `noise` | **4548** | 0.89 | 1.20 |

Each target explodes at exactly the endpoint its `to_velocity` divides
by zero at -- `beta(1)=0` for x_data, `alpha(0)=0` for noise -- and
velocity, which inverts nothing, stays flat. That is the diagnostic
doing its job.

## Tests

```bash
pytest -q     # ~2s, no dataset download
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
5. Swap the UNet for a Transformer (`--model jit`) and see which
   targets survive large patches -- see "A Transformer instead of the
   UNet (JiT)".
