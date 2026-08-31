"""dfm -- flow matching, built from scratch, factored the EDM way.

A generative process in this family is three independent choices, after
Karras et al. 2022 ("Elucidating the Design Space of Diffusion-Based
Generative Models"). Keeping them independent is the point of the
package: it turns "DDPM vs flow matching" from two codebases into two
rows of a table, and makes controlled comparisons possible.

    PATH     dfm.paths     how noise and data are interpolated
                           x_t = alpha(t)*x_data + sigma(t)*x_noise
    TARGET   dfm.targets   what the network predicts at (x_t, t)
                           velocity / x_data / x_noise -- interconvertible
    SAMPLER  dfm.samplers  how the learned field is integrated
                           dx/dt = v_theta(x, t), from t=0 to t=1

with two supporting axes that vary independently of all three:

    MODEL    dfm.mlp (2D), dfm.unet (images) -- both forward(x, t)
    LOSS     dfm.losses    t-distribution and per-timestep weighting

Currently implemented: LinearPath + VelocityTarget = rectified flow.
Euler and Heun samplers. Everything else is a slot with the derivation
written into the docstring where it goes.

Convention: t = 0 is NOISE, t = 1 is DATA, everywhere, and models
always take t as a float tensor in [0, 1] -- never a raw step index.
"""

from .losses import interpolant_loss, logit_normal_t, uniform_t
from .paths import LinearPath, Path
from .samplers import euler, heun
from .targets import Target, VelocityTarget

__version__ = "0.2.0"

__all__ = [
    "Path", "LinearPath",
    "Target", "VelocityTarget",
    "euler", "heun",
    "interpolant_loss", "uniform_t", "logit_normal_t",
]
