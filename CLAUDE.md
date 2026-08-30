# Project context for Claude sessions

Diffusion + flow matching models built from scratch for learning, with
Fashion-MNIST as the first dataset. See README.md for the full
picture; this file is conventions for anyone (human or Claude) editing
the code.

## Core design invariant

Every "process" (currently `dfm.ddpm.DDPM`, `dfm.flow_matching.RectifiedFlow`)
must expose exactly:

```python
process.training_loss(model, x1) -> torch.Tensor  # scalar
process.sample(model, shape, device, progress=True, **kwargs) -> torch.Tensor
```

and every model must accept `(x, t)` with `t` a float tensor in
`[0, 1]` (never a raw integer timestep). Keeping this contract is what
lets `dfm.trainer.Trainer` and the CLI scripts stay agnostic to which
process/model they're driving. When adding a new idea from the
literature, prefer adding a new process class or a new model over
branching inside an existing one.

## Where a new idea usually goes

- New corruption path / schedule -> new method or new class in
  `ddpm.py` / `flow_matching.py`, or a new file if it's a genuinely
  different process (e.g. `consistency.py`).
- New training target (v-prediction, EDM preconditioning, ...) ->
  change inside a process's `training_loss`, keep the signature.
- New sampler -> new method on the relevant process class
  (`DDPM.ddim_sample` is the existing example), or a standalone
  function called from a process's `sample`.
- New model architecture -> new file in `src/dfm/`, must implement
  `forward(x, t) -> same-shape tensor`.
- Conditioning (class labels, text, ...) -> extend the model's
  `forward` signature and thread the condition through
  `training_loss`/`sample`; keep the unconditional path working via a
  default (e.g. `None` = drop conditioning, for classifier-free
  guidance).

## Environment

- Package: `src/dfm/`, installed editable (`pip install -e .`).
- Dependencies: `requirements.txt` / `pyproject.toml`. Torch is CPU by
  default here; install from the CUDA index on a GPU machine (see the
  comment at the top of `requirements.txt`).
- Tests: `pytest -q` -- fast, no dataset download, just shape/numerics
  sanity checks (`tests/test_shapes.py`). Run these after any change to
  `unet.py`, `ddpm.py`, `flow_matching.py`, or `trainer.py`.
- Data: `dfm.data.get_fashion_mnist` downloads Fashion-MNIST into
  `data/` on first use (gitignored).
- Real training runs belong on a GPU machine, not this sandbox --
  `python scripts/train.py --max-steps N --subset M` is the pattern
  for a fast correctness smoke test instead.

## Style

- Small, readable modules over cleverness -- this repo is for building
  understanding, not for being maximally terse.
- Type hints on public functions/methods.
- Every process/schedule/sampler addition should get at least one
  `tests/test_shapes.py`-style shape/finiteness test.
