"""H3's visual caching: stills and clip frames through the video VAE, one sample per still or clip segment."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.data.media import read_video_frames
from maltrainer.data.minimax_dataset import MIN_CLIP_FRAMES, MediaSource, frames_tensor, load_image, plan_segments
from maltrainer.models.minimax_h3.video_vae import load_minimax_h3_video_vae, video_latent_frames
from maltrainer.trainers.minimax_h3.sample import H3Sample, MediaSettings
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)

PendingAudio = list[tuple[int, MediaSource, int]]  # (sample index, source, start frame): a clip's soundtrack to encode


def cache_visual(
    sources: list[MediaSource],
    vae_path: str,
    settings: MediaSettings,
    out: tuple[list[H3Sample], PendingAudio],
    device: torch.device,
    say: Callable[[str], None],
    show: bool,
) -> None:
    """Append a sample per still and per clip segment to ``out[0]``; clips with sound go to ``out[1]``."""
    samples = out[0]
    say(f"Loading Video VAE from {vae_path} for pre-caching...")
    vae = load_minimax_h3_video_vae(vae_path, device=device)
    for source in tqdm(sources, desc="Caching video latents", disable=not show):
        try:
            if source.kind == "image":
                pixels = load_image(source.path, settings.resolution, settings.aspect_mode).unsqueeze(0)
                with torch.no_grad():
                    latent = vae.encode_image(pixels.to(device)).detach().cpu()
                samples.append(H3Sample(str(source.path), "image", source.caption, 1, latent))
                continue
            _cache_clip(vae, source, settings, out, device)
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            logger.warning("Skipping %s: %s", source.path.name, e)
    del vae
    free_memory()
    say("Video VAE latents pre-cached and VAE unloaded from VRAM.")


def _cache_clip(
    vae: Any,
    source: MediaSource,
    settings: MediaSettings,
    out: tuple[list[H3Sample], PendingAudio],
    device: torch.device,
) -> None:
    samples, pending_audio = out
    frames = read_video_frames(source.path, settings.clip_resolution, settings.frame_budget)
    segments = plan_segments(len(frames), settings.max_frames, settings.long_media, MIN_CLIP_FRAMES)
    if not segments:
        logger.warning("Skipping %s: %d frames is shorter than one H3 clip", source.path.name, len(frames))
        return
    for start, count in segments:
        with torch.no_grad():
            latent = vae.encode_video(frames_tensor(frames[start : start + count]).to(device))
        latent = latent.detach().cpu()
        if latent.shape[2] != video_latent_frames(count):
            raise ValueError(
                f"{source.path.name}: {count} frames encoded to {latent.shape[2]} latent frames, "
                f"expected {video_latent_frames(count)}"
            )
        key = f"{source.path}@{start}" if len(segments) > 1 else str(source.path)
        samples.append(H3Sample(key, "clip", source.caption, count, latent))
        if settings.clip_audio and not source.muted:
            pending_audio.append((len(samples) - 1, source, start))
