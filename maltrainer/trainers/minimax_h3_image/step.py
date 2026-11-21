"""One H3 Image training step: a distilled noise level, the noised latent, and a plain flow-matching loss."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from maltrainer.models.minimax_h3.packing import patchify_video
from maltrainer.trainers.minimax_h3.sample import H3Sample


class ImageStepHost(Protocol):
    """What the step reads from the trainer."""

    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    text_cache: dict[str, torch.Tensor]
    sigmas: tuple[float, ...]

    def _draw_level(self) -> int: ...
    def _build_packed_layout(
        self, num_text_tokens: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
    ) -> dict[str, torch.Tensor]: ...
    def _forward_image(
        self, video_patches: Any, text_embeds: Any, sigma: float, layout: dict[str, Any], level: int
    ) -> Any: ...


def training_step(host: ImageStepHost, sample: H3Sample) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for one cached still. The checkpoint predicts ``x0 - noise`` (H3's convention) through its head for the
    level: x_t = (1 - sigma) * x0 + sigma * noise."""
    latents = sample.video.to(device=host.device, dtype=torch.float32)  # [1, 24, 1, h, w]
    _, _, frames, height, width = latents.shape
    caption = sample.caption
    dropout = float(host.config["train"].get("caption_dropout", 0.0) or 0.0)
    if dropout > 0 and torch.rand(()).item() < dropout:
        caption = ""
    text_embeds = host.text_cache[caption].to(device=host.device, dtype=host.dtype)

    level = host._draw_level()
    sigma = host.sigmas[level]
    noise = torch.randn_like(latents)
    noisy = (1.0 - sigma) * latents + sigma * noise
    layout = host._build_packed_layout(text_embeds.shape[1], frames, height, width, 0)
    output = host._forward_image(patchify_video(noisy).to(host.dtype), text_embeds, sigma, layout, level)
    loss = torch.nn.functional.mse_loss(output.float(), patchify_video(latents - noise).float())
    return loss, {"sigma": sigma, "video": float(loss.detach())}
