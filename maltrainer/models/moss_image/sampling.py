"""
MOSS-Image sampling for training previews, after the reference pipeline
(OpenMOSS-Team/MOSS-Image-Base pipeline_moss_image.py: MOSSImageScheduler + __call__), as Fizgig
ports it (fizgig/moss_image/sampling.py, Apache-2.0).

- start: fp32 noise [1, 128, h, w] from a CPU generator seeded with the seed (the reference's
  randn_tensor), so a seed renders the same picture every time
- schedule: ``steps`` sigmas on the shift-5 curve, each snapped to the nearest of the 1000
  training sigmas (j / 1000), repeats dropped, then 0
- Euler on the velocity v = noise - x0, in fp32: x <- x + (sigma_next - sigma) v
- CFG ``cfg`` against the negative prompt (empty by default), whose branch skips block 22

ComfyUI's scheduler does not snap to the training grid, so a seed renders slightly differently
there. Our DiT takes sigma in [0, 1] (it embeds 1000 sigma itself).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

SHIFT = 5.0  # scheduler_config.json shift
TRAIN_STEPS = 1000  # num_train_timesteps
UNCOND_SKIP = (22,)  # pipeline __call__ uncond_skip_layers default
DOWNSAMPLING = 16  # 8x VAE x 2x2 packing


def _grid(count: int, shift: float, total: int = TRAIN_STEPS) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def schedule(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """MOSSImageScheduler.set_timesteps plus the terminal 0: [n + 1] fp32, n <= steps (snapping can
    merge neighbours near sigma 1)."""
    if steps < 1 or shift <= 0:
        raise ValueError("steps and shift must be positive")
    sigmas = _grid(steps, shift)
    training = _grid(TRAIN_STEPS, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=TRAIN_STEPS - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.cat([torch.unique_consecutive(training[nearest]).float(), torch.zeros(1)])


def initial_noise(seed: int, height: int, width: int) -> torch.Tensor:
    if height % DOWNSAMPLING or width % DOWNSAMPLING:
        raise ValueError("height and width must be multiples of 16")
    generator = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn(
        (1, 128, height // DOWNSAMPLING, width // DOWNSAMPLING), generator=generator, dtype=torch.float32
    )


@torch.no_grad()
def sample(
    dit: Callable,
    context: torch.Tensor,
    height: int,
    width: int,
    *,
    steps: int = 30,
    seed: int = 42,
    cfg: float = 3.0,
    negative: Optional[torch.Tensor] = None,
    shift: float = SHIFT,
    uncond_skip: Sequence[int] = UNCOND_SKIP,
    device: Optional[torch.device] = None,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Denoised latents [1, 128, h, w] fp32. ``dit(x, sigma [1], context, skip_layers=...)`` is the
    DiT; context / negative: [L, 4096] or [1, L, 4096] text states. ``on_step(done, total)`` is
    called after every step (a progress bar)."""
    device = device or context.device
    x = initial_noise(seed, height, width).to(device)
    pos = (context if context.dim() == 3 else context[None]).to(device)
    use_cfg = cfg != 1.0 and negative is not None
    neg = (negative if negative.dim() == 3 else negative[None]).to(device) if use_cfg else None
    sigmas = schedule(steps, shift)
    total = len(sigmas) - 1
    for i in range(total):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        sigma = torch.tensor([s], device=device)
        v = dit(x, sigma, pos).float()
        if use_cfg:
            vu = dit(x, sigma, neg, skip_layers=tuple(uncond_skip)).float()
            v = vu + cfg * (v - vu)  # CFG combines fp32 predictions, as the reference
        x = x + (s_next - s) * v
        if on_step is not None:
            on_step(i + 1, total)
    return x
