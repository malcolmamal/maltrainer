"""Visage-Image sampling for training previews: the model's own recipe, an SDE-churn Euler with CFG.

Convention: x_t = (1 - t) x0 + t noise, the DiT predicts noise - x0, and sampling runs from t = 1 to t = 0 over a grid
warped toward the noisy end. Before each step the "sde" solver re-noises the sample from t up to t (1 + churn), exactly
for this interpolant, then steps from there; "euler" is the plain ODE. Guidance is ``uncond + cfg * (cond - uncond)``
against a zero caption, which is what training dropped captions to. Defaults are the released recipe (sde, churn 0.3,
cfg 3, 50 steps, shift 3.3); steps and churn are coupled, since churn is applied per step.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import torch

DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_CHURN, DEFAULT_SHIFT = 50, 3.0, 0.3, 3.3
SOLVERS = ("sde", "euler")
SDE_T_MAX = 0.9


def shifted_timesteps(steps: int, shift: float) -> torch.Tensor:
    """The ``steps + 1`` times from 1 to 0, uniform and then warped: t' = s t / (1 + (s - 1) t)."""
    t = torch.linspace(1.0, 0.0, steps + 1)
    return shift * t / (1.0 + (shift - 1.0) * t)


def churned(x: torch.Tensor, t: float, churn: float, noise: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``x`` re-noised from ``t`` up to min(t (1 + churn), 0.9), and that new time."""
    t_hat = min(t * (1 + churn), SDE_T_MAX)
    scale = (1 - t_hat) / (1 - t)
    variance = max(t_hat**2 - (scale * t) ** 2, 0.0)
    return scale * x + variance**0.5 * noise, t_hat


@torch.no_grad()
def sample_latents(
    transformer: Any,
    cond: torch.Tensor,
    grid: tuple[int, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    solver: str = "sde",
    churn: float = DEFAULT_CHURN,
    shift: float = DEFAULT_SHIFT,
    on_step: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Denoised latents [1, C, h, w] for ``grid`` = (h, w); ``cond`` is the caption's [1, L, D] states.
    ``on_step(done, total)`` is called after every step."""
    if solver not in SOLVERS:
        raise ValueError(f"sample.solver must be one of {SOLVERS}, got {solver!r}")
    channels = transformer.config.in_channels
    x = torch.randn((1, channels, *grid), generator=torch.Generator().manual_seed(seed)).to(device)
    noise_generator = torch.Generator().manual_seed(seed + 1)
    text = torch.cat([cond, torch.zeros_like(cond)]).to(device, dtype) if guidance != 1.0 else cond.to(device, dtype)
    times = shifted_timesteps(steps, shift).tolist()
    for index, (t_cur, t_next) in enumerate(pairwise(times), 1):
        t = t_cur
        if solver == "sde" and churn > 0 and 0 < t_cur < SDE_T_MAX:
            x, t = churned(x, t_cur, churn, torch.randn(x.shape, generator=noise_generator).to(device))
        x_in = torch.cat([x, x]) if guidance != 1.0 else x
        velocity = transformer(x_in.to(dtype), torch.full((x_in.shape[0], 1), t, device=device), text).float()
        if guidance != 1.0:
            conditional, plain = velocity.chunk(2)
            velocity = plain + guidance * (conditional - plain)
        x = x + (t_next - t) * velocity
        if on_step is not None:
            on_step(index, steps)
    return x
