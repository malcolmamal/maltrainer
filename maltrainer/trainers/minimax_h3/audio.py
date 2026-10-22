"""H3's audio caching: clip soundtracks and audio-only recordings through the audio VAE, as packed stereo rows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.data.media import read_audio
from maltrainer.data.minimax_dataset import (
    AUDIO_SAMPLE_RATE,
    FPS,
    MIN_AUDIO_FRAMES,
    MediaSource,
    plan_segments,
    slice_audio,
)
from maltrainer.models.minimax_h3.audio_vae import load_minimax_h3_audio_encoder
from maltrainer.models.minimax_h3.packing import pack_audio_rows
from maltrainer.models.minimax_h3.video_vae import video_latent_frames
from maltrainer.trainers.minimax_h3.sample import H3Sample, MediaSettings, check_audio_rows
from maltrainer.trainers.minimax_h3.visual import PendingAudio
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)

Encode = Callable[[Any], torch.Tensor]  # a waveform slice -> its packed audio rows


def cache_audio(
    model_config: dict[str, Any],
    work: tuple[PendingAudio, list[MediaSource]],
    settings: MediaSettings,
    samples: list[H3Sample],
    device: torch.device,
    say: Callable[[str], None],
    show: bool,
) -> None:
    """The soundtracks of the clips ``work[0]`` names onto their samples; a sample per segment of each audio file
    in ``work[1]`` (a zeros video placeholder, no video loss)."""
    pending_audio, audio_sources = work
    audio_vae_path = model_config.get("audio_vae_path")
    if not audio_vae_path:
        raise ValueError(
            "model.audio_vae_path is required to train on clip soundtracks or audio files "
            "(ComfyUI's minimax_h3_audio_vae_fp32.safetensors). Set train.train_audio: false "
            "to train video only."
        )
    say(f"Loading Audio VAE from {audio_vae_path} for pre-caching...")
    audio_vae = load_minimax_h3_audio_encoder(audio_vae_path, device=device)
    encode = partial(_encode, audio_vae, device)
    _clip_soundtracks(pending_audio, samples, encode, show)
    budget = settings.frame_budget
    for source in tqdm(audio_sources, desc="Caching audio items", disable=not show):
        try:
            _audio_item(source, (settings, budget), samples, encode)
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            logger.warning("Skipping %s: %s", source.path.name, e)
    del encode, audio_vae
    free_memory()
    say("Audio latents pre-cached and Audio VAE unloaded from VRAM.")


def _encode(audio_vae: Any, device: torch.device, wav: Any) -> torch.Tensor:
    waveform = torch.from_numpy(wav).unsqueeze(0).to(device)
    return pack_audio_rows(audio_vae.encode(waveform)).detach().cpu()


def _clip_soundtracks(pending_audio: PendingAudio, samples: list[H3Sample], encode: Encode, show: bool) -> None:
    decoded: dict[str, Any] = {}
    for index, source, start in tqdm(pending_audio, desc="Caching clip audio", disable=not show):
        key = str(source.path)
        if key not in decoded:
            decoded = {key: read_audio(source.path)}  # one file at a time in memory
        wav = decoded[key]
        if wav is None:
            continue  # no track / digital silence: trains as a muted clip
        sample = samples[index]
        rows = encode(slice_audio(wav, start, sample.frames))
        check_audio_rows(sample, rows)
        sample.audio = rows
    decoded.clear()


def _audio_item(
    source: MediaSource, frames: tuple[MediaSettings, int], samples: list[H3Sample], encode: Encode
) -> None:
    settings, budget = frames
    wav = read_audio(source.path)
    if wav is None:
        logger.warning("Skipping %s: no decodable sound (or digital silence)", source.path.name)
        return
    total = min(int(wav.shape[1] / AUDIO_SAMPLE_RATE * FPS), budget)
    segments = plan_segments(total, settings.max_frames, settings.long_media, MIN_AUDIO_FRAMES)
    if not segments:
        logger.warning("Skipping %s: shorter than %.2f s", source.path.name, MIN_AUDIO_FRAMES / FPS)
        return
    for start, count in segments:
        # A zeros video latent is the dataset mean in normalized space. It sizes
        # the packed sequence and gives the audio rows something to attend
        # across; the video loss is dropped for these items.
        side = max(32, settings.placeholder_px // 32 * 32) // 16
        placeholder = torch.zeros(1, 24, video_latent_frames(count), side, side)
        key = f"{source.path}@{start}" if len(segments) > 1 else str(source.path)
        sample = H3Sample(key, "audio", source.caption, count, placeholder, audio_only=True)
        sample.audio = encode(slice_audio(wav, start, count))
        check_audio_rows(sample, sample.audio)
        samples.append(sample)
