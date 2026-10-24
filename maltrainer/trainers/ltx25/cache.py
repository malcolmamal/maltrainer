"""LTX-2.5's caching: video latents (and soundtracks), audio latents and caption features, each encoder loaded once
and freed before the next, so the 22B DiT never shares the GPU with them."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from maltrainer.data.minimax_dataset import MediaSource
from maltrainer.models.ltx25.audio_vae import load_audio_vae_encoder
from maltrainer.models.ltx25.video_vae import load_video_vae_encoder
from maltrainer.trainers.ltx25.data import MediaSettings, audio_latent_count, load_pixels
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


@dataclass
class MediaCache:
    video: dict[str, torch.Tensor] = field(default_factory=dict)  # path -> [128, F', H', W']
    audio: dict[str, torch.Tensor] = field(default_factory=dict)  # path -> [8, T, 16]
    frames: dict[str, int] = field(default_factory=dict)  # path -> pixel frames of the clip


def cache_video(
    sources: list[MediaSource], model_config: dict[str, Any], settings: MediaSettings, device: torch.device, show: bool
) -> tuple[MediaCache, dict[str, np.ndarray]]:
    """Every source's latent; the soundtrack windows (still waveforms) of the clips that have one."""
    cache, soundtracks = MediaCache(), {}
    dtype = getattr(torch, str(model_config.get("vae_dtype", "bfloat16")))
    vae = load_video_vae_encoder(model_config["video_vae_path"], device, dtype)
    for source in tqdm(sources, desc="Caching latents", disable=not show):
        key = str(source.path)
        try:
            pixels, wav = load_pixels(source, settings)
            cache.video[key] = vae.encode(pixels.to(device, dtype))[0].float().cpu()
        except Exception as e:  # a file PyAV or PIL cannot decode skips that sample; a missing PyAV stops the run
            if isinstance(e, ImportError):
                raise
            logger.warning("Skipping %s: %s", source.path.name, e)
            continue
        cache.frames[key] = pixels.shape[2]
        if wav is not None:
            soundtracks[key] = wav
    del vae
    free_memory()
    return cache, soundtracks


def cache_audio(
    cache: MediaCache,
    soundtracks: dict[str, np.ndarray],
    model_config: dict[str, Any],
    fps: float,
    device: torch.device,
    show: bool,
) -> None:
    if not soundtracks:
        return
    path = model_config.get("audio_vae_path")
    if not path:
        raise ValueError(
            "model.audio_vae_path is required to train on clip soundtracks (set dataset.audio: false for video only)"
        )
    vae = load_audio_vae_encoder(path, device)
    for key, wav in tqdm(soundtracks.items(), desc="Caching audio", disable=not show):
        latent = vae.encode(torch.from_numpy(wav).unsqueeze(0).to(device), 16000)[0].cpu()
        expected = audio_latent_count(cache.frames[key], fps)
        if latent.shape[1] != expected:
            logger.warning(
                "Skipping the soundtrack of %s: %d audio latents, expected %d", key, latent.shape[1], expected
            )
            continue
        cache.audio[key] = latent
    del vae
    free_memory()
