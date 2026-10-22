"""The Z-Image textual-inversion trainer's step: the learned token's vectors come from the text encoder's input
embeddings, so the caption features keep their gradient; everything else in the loss is the LoRA trainer's."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

import torch

from maltrainer.trainers.zimage.prediction import prediction
from maltrainer.trainers.zimage.step import ZIMAGE_SCALE, ZIMAGE_SHIFT
from maltrainer.trainers.zimage.text import chat_prompts, tokens


class EmbeddingHost(Protocol):
    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    vae: Any
    tokenizer: Any
    text_encoder: Any
    transformer: Any


def batch_loss(host: EmbeddingHost, batch: dict[str, Any]) -> torch.Tensor:
    """Plain flow-matching MSE (uniform timesteps) with gradients through the text encoder's embeddings."""
    with torch.no_grad():
        latents = host.vae.encode(batch["pixel_values"].to(host.device)).latent_dist.sample()
        latents = ((latents - ZIMAGE_SHIFT) * ZIMAGE_SCALE).to(dtype=host.dtype)
    batch_tokens = tokens(host.tokenizer, chat_prompts(host.tokenizer, batch["caption"], thinking=True), host.device)
    base_text_encoder = getattr(host.text_encoder, "model", host.text_encoder)
    te_out = base_text_encoder(
        batch_tokens.input_ids, attention_mask=batch_tokens.attention_mask, output_hidden_states=True
    )
    text_embeds = te_out.hidden_states[-2]

    noise = torch.randn_like(latents)
    timesteps = torch.rand(latents.shape[0], device=host.device)
    noisy_latents = (1 - timesteps[:, None, None, None]) * latents + timesteps[:, None, None, None] * noise
    noisy_latents = noisy_latents.to(dtype=host.dtype)

    takes_mask = "cap_mask" in inspect.signature(host.transformer.forward).parameters
    kwargs = {"cap_mask": batch_tokens.attention_mask.bool()} if takes_mask else {}
    output = host.transformer(noisy_latents.unsqueeze(2), 1.0 - timesteps, text_embeds, **kwargs)
    model_pred = prediction(output, latents, stack=True)
    # Z-Image / Musubi uses target = latents - noise
    return torch.nn.functional.mse_loss(model_pred.float(), (latents - noise).float(), reduction="mean")
