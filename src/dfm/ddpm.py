"""DDPM: discrete-time Gaussian diffusion (Ho, Jain & Abbeel 2020).

Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
Training target: predict the noise eps that was added (eps-prediction).
Sampling: ancestral sampling, i.e. the standard T-step reverse chain,
plus a DDIM sampler for fewer, deterministic steps.

This is one interchangeable "process" -- see dfm/flow_matching.py for
the other one. Both expose `.training_loss(model, x1)` and
`.sample(model, shape, device, ...)` so dfm/trainer.py can drive
either without knowing which it has.

Extension points if you want to try ideas from the literature:
  - swap `beta_schedule` for a new one (e.g. sigmoid, a learned schedule)
  - change the training target to v-prediction (see NOTE in training_loss)
  - replace `ddim_sample` with a smarter solver (DPM-Solver, PLMS, ...)
  - add classifier-free guidance by making `model` take a class label
    and blending conditional/unconditional predictions in the sampler
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021) cosine schedule -- corrupts more gently
    at the start/end than the linear schedule, generally better for
    small images like Fashion-MNIST."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(max=0.999)


SCHEDULES = {"linear": linear_beta_schedule, "cosine": cosine_beta_schedule}


def _extract(a: torch.Tensor, t_idx: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather per-timestep scalars in `a` at indices `t_idx`, reshaped
    to broadcast against a batch of images `shape` = (B, C, H, W)."""
    out = a.to(t_idx.device).gather(0, t_idx)
    return out.reshape(t_idx.shape[0], *([1] * (len(shape) - 1)))


class DDPM:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        self.timesteps = timesteps
        betas = SCHEDULES[schedule](timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt()
        # posterior q(x_{t-1} | x_t, x_0) variance, Ho et al. eq. 7
        self.posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _extract(self.sqrt_alphas_bar, t_idx, x0.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alphas_bar, t_idx, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def training_loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t_idx = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t_idx, noise)
        t_norm = t_idx.float() / (self.timesteps - 1)
        pred_noise = model(x_t, t_norm)
        # NOTE: this is eps-prediction. To try v-prediction instead,
        # change the target to `sqrt_ab * noise - sqrt_1mab * x0` and
        # convert predictions back to eps/x0 accordingly at sample time.
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, shape, device, progress: bool = True) -> torch.Tensor:
        """Full T-step ancestral sampling (the original DDPM sampler)."""
        B = shape[0]
        x = torch.randn(shape, device=device)
        iterator = reversed(range(self.timesteps))
        if progress:
            iterator = tqdm(iterator, total=self.timesteps, desc="DDPM sampling")
        for t in iterator:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = t_batch.float() / (self.timesteps - 1)
            eps = model(x, t_norm)

            beta_t = self.betas[t].to(device)
            alpha_t = self.alphas[t].to(device)
            sqrt_1mab_t = self.sqrt_one_minus_alphas_bar[t].to(device)

            mean = (x - beta_t / sqrt_1mab_t * eps) / alpha_t.sqrt()
            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.posterior_variance[t].to(device).sqrt()
                x = mean + sigma * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps: int = 50, eta: float = 0.0, progress: bool = True) -> torch.Tensor:
        """DDIM sampler (Song, Meng & Ermon 2021): a deterministic (eta=0)
        or partially-stochastic (0 < eta <= 1) sampler that skips steps,
        trading a bit of quality for a large speedup over full ancestral
        sampling. Useful once you want fast iteration on ideas."""
        B = shape[0]
        step_indices = torch.linspace(0, self.timesteps - 1, steps, device=device).long()
        step_indices = torch.unique(step_indices, sorted=True)
        x = torch.randn(shape, device=device)
        iterator = reversed(range(len(step_indices)))
        if progress:
            iterator = tqdm(iterator, total=len(step_indices), desc="DDIM sampling")
        for i in iterator:
            t = step_indices[i].item()
            t_prev = step_indices[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = t_batch.float() / (self.timesteps - 1)
            eps = model(x, t_norm)

            ab_t = self.alphas_bar[t].to(device)
            ab_prev = self.alphas_bar[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)).sqrt() if t_prev >= 0 else torch.tensor(0.0, device=device)
            dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = torch.randn_like(x) if (eta > 0 and t_prev >= 0) else 0.0
            x = ab_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x
