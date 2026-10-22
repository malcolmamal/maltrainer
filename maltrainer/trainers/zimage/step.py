"""One Z-Image LoRA batch: VAE latents and caption features computed on the fly, flow-matching noise, the
transformer's prediction (optionally pushed away from an unconditional one), the weighted loss, and the optimizer
step, progress line and due save that follow."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol

import torch

from maltrainer.trainers.zimage.loss import weighted_loss
from maltrainer.trainers.zimage.noise import transformed
from maltrainer.trainers.zimage.prediction import prediction
from maltrainer.trainers.zimage.text import caption_features, chat_prompts, tokens

# Z-Image's latent shift and scale (Musubi-Tuner's zimage_config.py), not the VAE config's scaling factor.
ZIMAGE_SHIFT = 0.1159
ZIMAGE_SCALE = 0.3611


class BatchHost(Protocol):
    """What the batch reads from the trainer (``_sample_timesteps`` stays the trainer's: tests call it)."""

    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    vae: Any
    tokenizer: Any
    text_encoder: Any
    transformer: Any
    lora_network: Any
    accelerator: Any
    optimizer: Any
    lr_scheduler: Any
    noise_scheduler: Any
    progress_logger: logging.Logger

    def _sample_timesteps(self, batch_size: int, num_train_timesteps: int) -> torch.Tensor: ...
    def save_checkpoint(self, step: int | str) -> None: ...


def batch_step(
    host: BatchHost, batch: dict[str, Any], epochs: tuple[int, int], steps: tuple[int, int], bar: Any
) -> int:
    """One batch's micro-step (and the optimizer step when the accumulation completes); returns the step."""
    train_config = host.config["train"]
    (epoch, num_epochs), (global_step, total_steps) = epochs, steps
    with host.accelerator.accumulate(host.lora_network):
        loss = batch_loss(host, batch, host.noise_scheduler.config.num_train_timesteps)
        host.accelerator.backward(loss)
        if host.accelerator.sync_gradients:
            host.accelerator.clip_grad_norm_(host.lora_network.parameters(), train_config.get("max_grad_norm", 1.0))
            host.optimizer.step()
            if host.lr_scheduler is not None:
                host.lr_scheduler.step()
            host.optimizer.zero_grad()
    if host.accelerator.sync_gradients:
        global_step += 1
        bar.update(1)
        lr = host.optimizer.param_groups[0]["lr"]
        host.progress_logger.info(
            f"Step: {global_step}/{total_steps} [{int(global_step / total_steps * 100)}%], "
            f"Loss: {loss.item():.6f}, Epoch: {epoch + 1}/{num_epochs}, LR: {lr:.2e}"
        )
        save_every = int(train_config.get("save_every", 500) or 0)
        if save_every > 0 and global_step % save_every == 0:
            host.save_checkpoint(global_step)
    bar.set_postfix({"loss": loss.item(), "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"})
    return global_step


def batch_loss(host: BatchHost, batch: dict[str, Any], num_train_timesteps: int) -> torch.Tensor:
    train_config = host.config["train"]
    captions = batch["caption"]
    with torch.no_grad():
        latents = host.vae.encode(batch["pixel_values"].to(host.device)).latent_dist.sample()
        latents = ((latents - ZIMAGE_SHIFT) * ZIMAGE_SCALE).to(dtype=host.dtype)
        cond_tokens = tokens(host.tokenizer, chat_prompts(host.tokenizer, captions, thinking=True), host.device)
    text_embeds = caption_features(host.text_encoder, cond_tokens)

    latents, noise = transformed(latents, torch.randn_like(latents), train_config)
    timesteps = host._sample_timesteps(latents.shape[0], num_train_timesteps)
    # flow matching: 0 is clean, 1 is noise
    noisy_latents = (1 - timesteps[:, None, None, None]) * latents + timesteps[:, None, None, None] * noise
    noisy_latent_multiplier = float(train_config.get("noisy_latent_multiplier", 1.0))
    if noisy_latent_multiplier != 1.0:
        noisy_latents = noisy_latents * noisy_latent_multiplier

    takes_mask = "cap_mask" in inspect.signature(host.transformer.forward).parameters
    model_pred = prediction(
        _forward(host, (noisy_latents, timesteps, text_embeds), cond_tokens, takes_mask), latents, stack=True
    )
    if train_config.get("do_differential_guidance", False):
        prompt = train_config.get("unconditional_prompt", "")
        uncond_tokens = tokens(
            host.tokenizer, chat_prompts(host.tokenizer, [prompt] * len(captions), thinking=False), host.device
        )
        uncond_embeds = caption_features(host.text_encoder, uncond_tokens)
        uncond = _forward(host, (noisy_latents, timesteps, uncond_embeds), uncond_tokens, takes_mask)
        uncond_pred = prediction(uncond, latents, stack=False)
        guidance_scale = float(train_config.get("differential_guidance_scale", 3.0))
        model_pred = uncond_pred + guidance_scale * (model_pred - uncond_pred)

    # Z-Image / Musubi uses target = latents - noise (standard flow matching is noise - latents)
    return weighted_loss(model_pred, latents - noise, timesteps, train_config)


def _forward(host: BatchHost, inputs: tuple[torch.Tensor, ...], batch_tokens: Any, takes_mask: bool) -> Any:
    """The LoRA'd transformer on (x, t, caption features). Z-Image takes t in [0, 1] with 1 clean (it scales by
    1000 itself), so it gets 1 - timesteps; cap_mask when its forward takes one."""
    noisy_latents, timesteps, embeds = inputs
    kwargs = {"cap_mask": batch_tokens.attention_mask.bool()} if takes_mask else {}
    return host.lora_network(noisy_latents.unsqueeze(2), 1.0 - timesteps, embeds, **kwargs)
