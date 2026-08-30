"""dfm: diffusion and flow matching models, built from scratch.

Design: one shared UNet backbone (dfm.unet.UNet) that always takes
(x_t, t_normalized in [0, 1]) and predicts either a noise vector (DDPM)
or a velocity vector (flow matching). Everything that differs between
"diffusion" and "flow matching" -- how corruption is defined, what the
training target is, how sampling integrates the model's predictions --
lives in a small, swappable "process" object:

    dfm.ddpm.DDPM              -- discrete-time Gaussian diffusion
    dfm.flow_matching.RectifiedFlow -- continuous-time flow matching

Both processes expose the same two methods so dfm.trainer.Trainer can
drive either one without caring which it is:

    process.training_loss(model, x1) -> scalar loss
    process.sample(model, shape, device, ...) -> generated batch

To try a new idea from the literature, you generally only need to
add or modify one of: a schedule, a process class, or a sampler
function -- see CLAUDE.md and README.md for the extension points.
"""

__version__ = "0.1.0"
