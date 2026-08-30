# %% [markdown]
# # DDPM vs. flow matching, side by side
#
# This is a "notebook" in the VS Code / Jupyter interactive-window sense:
# the `# %%` markers delimit cells you can run one at a time (VS Code's
# Python extension and PyCharm both understand this natively; if you'd
# rather have a real .ipynb, `pip install jupytext` and run
# `jupytext --to notebook 01_explore_ddpm_vs_flow_matching.py`).
#
# Goal: build intuition for what each process's forward corruption
# looks like, before/instead of committing to a long training run.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] / "src"))
if not (Path.cwd().parents[0] / "src").exists():
    sys.path.insert(0, str(Path.cwd() / "src"))  # if run from repo root instead

import matplotlib.pyplot as plt
import torch

from dfm.data import get_fashion_mnist
from dfm.ddpm import DDPM
from dfm.flow_matching import RectifiedFlow

torch.manual_seed(0)

# %% [markdown]
# ## Load a couple of real images

# %%
ds = get_fashion_mnist(train=True)
imgs = torch.stack([ds[i][0] for i in range(4)])  # (4, 1, 28, 28), already in [-1, 1]

# %% [markdown]
# ## DDPM forward process: q(x_t | x_0)
#
# Watch how a cosine schedule corrupts the image at increasing t.
# Notice it stays *recognizable* for longer than a linear schedule would
# (try swapping `schedule="cosine"` for `"linear"` and re-running).

# %%
ddpm = DDPM(timesteps=1000, schedule="cosine")
ts = [0, 100, 300, 600, 999]

fig, axes = plt.subplots(len(imgs), len(ts), figsize=(2 * len(ts), 2 * len(imgs)))
for row, img in enumerate(imgs):
    for col, t in enumerate(ts):
        t_idx = torch.full((1,), t, dtype=torch.long)
        x_t = ddpm.q_sample(img.unsqueeze(0), t_idx)[0, 0]
        axes[row, col].imshow(x_t, cmap="gray", vmin=-1, vmax=1)
        axes[row, col].set_title(f"t={t}")
        axes[row, col].axis("off")
fig.suptitle("DDPM forward corruption (cosine schedule)")
fig.tight_layout()
fig.savefig("ddpm_forward_process.png", dpi=120)
print("saved ddpm_forward_process.png")

# %% [markdown]
# ## Flow matching's path: a straight line between noise and data
#
# Same idea, but t is continuous in [0, 1] and the path is literally
# `x_t = (1-t) * x0 + t * x1` -- a straight line in pixel space, no
# schedule to tune. Compare how "linear-looking" this corruption is
# versus DDPM's, especially in the middle of the range.

# %%
rf = RectifiedFlow()
ts_continuous = [0.0, 0.1, 0.3, 0.6, 1.0]

fig, axes = plt.subplots(len(imgs), len(ts_continuous), figsize=(2 * len(ts_continuous), 2 * len(imgs)))
for row, img in enumerate(imgs):
    x1 = img.unsqueeze(0)
    x0 = torch.randn_like(x1)
    for col, t in enumerate(ts_continuous):
        x_t = (1 - t) * x0 + t * x1
        axes[row, col].imshow(x_t[0, 0], cmap="gray", vmin=-1, vmax=1)
        axes[row, col].set_title(f"t={t}")
        axes[row, col].axis("off")
fig.suptitle("Flow matching forward path (t=0 noise -> t=1 data)")
fig.tight_layout()
fig.savefig("flow_matching_forward_process.png", dpi=120)
print("saved flow_matching_forward_process.png")

# %% [markdown]
# ## Next: train both
#
# ```bash
# python scripts/train.py --process ddpm --epochs 20
# python scripts/train.py --process flow_matching --epochs 20
# ```
#
# then generate a grid from a checkpoint:
#
# ```bash
# python scripts/sample.py --checkpoint runs/ddpm/checkpoint.pt --process ddpm --sampler ddim --steps 50 --out ddpm_grid.png
# python scripts/sample.py --checkpoint runs/flow_matching/checkpoint.pt --process flow_matching --steps 50 --out fm_grid.png
# ```
