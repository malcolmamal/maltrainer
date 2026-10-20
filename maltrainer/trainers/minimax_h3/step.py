"""One H3 training step: the noised video and audio streams, the optional guidance targets, and the loss."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from maltrainer.models.minimax_h3.packing import apply_flow_shift, patchify_video, remap_sigma
from maltrainer.trainers.minimax_h3.sample import H3Sample


class StepHost(Protocol):
    """What the step reads from the trainer (its methods stay the trainer's, where tests patch them)."""

    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    text_cache: dict[str, torch.Tensor]

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor: ...
    def _train_audio(self) -> bool: ...
    def _do_guidance_loss(self) -> bool: ...
    def _build_packed_layout(
        self, num_text_tokens: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
    ) -> dict[str, torch.Tensor]: ...
    def _forward(
        self, video_patches: Any, audio_rows: Any, text_embeds: Any, timestep: Any, layout: dict[str, Any]
    ) -> Any: ...


def guided_target(
    target: torch.Tensor, uncond: torch.Tensor, sigma: torch.Tensor, scale: float, schedule: str, cfg_zero: bool
) -> torch.Tensor:
    """Contrastive guidance target for one stream: uncond + scale * (target - uncond).

    H3 is guidance-distilled: CFG is baked into the weights, so a plain flow-matching
    target trains the conditional direction too weakly and the distillation degrades as
    the LoRA takes hold. The fix extrapolates the TARGET away from the model's own
    (detached) unconditional prediction, rather than applying CFG to the prediction.
    Each stream uses its own sigma for the schedule, so audio decays on the audio clock.
    """
    uncond = uncond.detach().float()
    guided = target.float()
    bsz = guided.shape[0]
    if cfg_zero:
        # CFG-Zero*: rescale the unconditional direction onto the conditional one before
        # extrapolating, so a poorly-aligned uncond prediction does not drag the target sideways.
        flat_cond = guided.reshape(bsz, -1)
        flat_uncond = uncond.reshape(bsz, -1)
        dot = (flat_cond * flat_uncond).sum(dim=1, keepdim=True)
        squared_norm = (flat_uncond**2).sum(dim=1, keepdim=True) + 1e-8
        uncond = uncond * (dot / squared_norm).view(bsz, 1, 1)
    # (target - uncond) carries sigma * fresh_noise, which nothing can predict at low
    # sigma, so the "sigma" schedule decays toward a plain flow target there.
    scale_eff = 1.0 + (scale - 1.0) * sigma.detach().float().view(bsz, 1, 1) if schedule == "sigma" else scale
    return (uncond + scale_eff * (guided - uncond)).detach().to(target.dtype)


def training_step(host: StepHost, sample: H3Sample, flow_shift: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for one cached sample. Returns (loss, parts for logging)."""
    train_config = host.config["train"]
    latents = sample.video.to(device=host.device, dtype=torch.float32)  # [1, 24, T, h, w]
    _, _, f_lat, h_lat, w_lat = latents.shape

    caption = sample.caption
    dropout = float(train_config.get("caption_dropout", 0.0) or 0.0)
    if dropout > 0 and torch.rand(()).item() < dropout:
        caption = ""
    text_embeds = host.text_cache[caption].to(device=host.device, dtype=host.dtype)

    # Sigmas: video from the configured schedule, audio on its coupled shift-3 curve.
    sigma_v = apply_flow_shift(host._sample_timesteps(1), shift=flow_shift)
    sigma_a = remap_sigma(sigma_v)

    # x_t = (1 - sigma) * x0 + sigma * noise; the H3 checkpoint predicts (x0 - noise).
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - sigma_v.view(1, 1, 1, 1, 1)) * latents + sigma_v.view(1, 1, 1, 1, 1) * noise
    video_patches = patchify_video(noisy_latents).to(host.dtype)
    targets = [patchify_video(latents - noise), None]  # video, audio

    audio_t, audio_rows, targets[1] = _audio_stream(host, sample, sigma_a)

    # H3's DiT is conditioned on t = 1 - sigma (comfy/ldm/minimax/model.py). Row 0 is the
    # video/text timestep, row 1 the audio one (see packing.packed_layout).
    timestep = torch.cat([1.0 - sigma_v, 1.0 - sigma_a]) if audio_t else 1.0 - sigma_v
    layout = host._build_packed_layout(text_embeds.shape[1], f_lat, h_lat, w_lat, audio_t)
    model_output = host._forward(video_patches, audio_rows, text_embeds, timestep, layout)

    train_video = not sample.audio_only
    if host._do_guidance_loss():
        uncond_prompt = str(train_config.get("unconditional_prompt", ""))
        uncond_embeds = host.text_cache[uncond_prompt].to(device=host.device, dtype=host.dtype)
        # The empty prompt is one BOS token where the caption is several, and the packed
        # layout is sized from the text length -- indices, tags and every media row's
        # RoPE origin shift with it. Build its own.
        uncond_layout = host._build_packed_layout(uncond_embeds.shape[1], f_lat, h_lat, w_lat, audio_t)
        with torch.no_grad():
            uncond_output = host._forward(video_patches, audio_rows, uncond_embeds, timestep, uncond_layout)
        _guide(targets, uncond_output, (sigma_v, sigma_a), train_config, train_video)
    return _loss(model_output, targets, train_video, train_config, float(sigma_v))


def _audio_stream(
    host: StepHost, sample: H3Sample, sigma_a: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor | None]:
    """(audio latent frames packed, the noised rows, their target or None).

    ComfyUI never runs H3 without audio rows, so even a still packs its 2-latent (0.083 s) stereo block:
    silence (x0 = 0) noised at the audio sigma, no loss. A sample with real sound gets a real target.
    pack_silent_audio: false reproduces the old audio-free layout for A/B comparisons.
    """
    use_audio = sample.audio is not None and host._train_audio()
    pack_audio = use_audio or bool(host.config["train"].get("pack_silent_audio", True))
    audio_t = sample.audio_latents if pack_audio else 0
    audio_target = None
    if use_audio and sample.audio is not None:
        a0 = sample.audio.to(device=host.device, dtype=torch.float32)
        a_noise = torch.randn_like(a0)
        audio_rows = (1.0 - sigma_a.view(1, 1, 1)) * a0 + sigma_a.view(1, 1, 1) * a_noise
        audio_target = a0 - a_noise
    else:
        audio_rows = sigma_a.view(1, 1, 1) * torch.randn(1, audio_t * 2, 32, device=host.device)
    return audio_t, audio_rows.to(host.dtype), audio_target


def _guide(
    targets: list[Any],
    uncond: Any,
    sigmas: tuple[torch.Tensor, torch.Tensor],
    train_config: dict[str, Any],
    video: bool,
) -> None:
    """Both streams' targets extrapolated away from the unconditional prediction (``guided_target``)."""
    scale = float(train_config.get("guidance_loss_target", 3.5))
    schedule = str(train_config.get("guidance_loss_schedule", "sigma")).lower()
    cfg_zero = bool(train_config.get("guidance_loss_cfg_zero", False))
    if video:
        targets[0] = guided_target(targets[0], uncond.video, sigmas[0], scale, schedule, cfg_zero)
    if targets[1] is not None:
        targets[1] = guided_target(targets[1], uncond.audio, sigmas[1], scale, schedule, cfg_zero)


def _loss(
    model_output: Any, targets: list[Any], train_video: bool, train_config: dict[str, Any], sigma: float
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, float] = {"sigma": sigma}
    loss = None
    if train_video:
        # Not 0 * loss for audio-only items: their video is a placeholder, and even a
        # zero-weighted term would build (and walk) the whole video backward graph.
        video_loss = torch.nn.functional.mse_loss(model_output.video.float(), targets[0].float())
        parts["video"] = float(video_loss.detach())
        loss = video_loss
    if targets[1] is not None:
        audio_loss = torch.nn.functional.mse_loss(model_output.audio.float(), targets[1].float())
        parts["audio"] = float(audio_loss.detach())
        weighted = float(train_config.get("audio_loss_weight", 1.0)) * audio_loss
        loss = weighted if loss is None else loss + weighted

    if loss is None:
        raise RuntimeError("no loss term for this sample")
    loss_weight_strength = float(train_config.get("loss_weight_strength", 1.0))
    if loss_weight_strength != 1.0:
        loss = loss * loss_weight_strength
    return loss, parts
