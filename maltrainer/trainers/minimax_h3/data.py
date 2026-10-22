"""H3's dataset, encoded once before the transformer loads: video latents, audio rows and caption embeddings,
each encoder unloaded before the next loads so peak VRAM stays at one model. Console lines go through ``say``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.data.minimax_dataset import MEDIA_KINDS, count_media_files, scan_media_sources, summarize_kinds
from maltrainer.models.minimax_h3.text_encoder import load_minimax_h3_text_encoder
from maltrainer.trainers.minimax_h3.audio import cache_audio
from maltrainer.trainers.minimax_h3.sample import H3Sample, MediaSettings
from maltrainer.trainers.minimax_h3.visual import PendingAudio, cache_visual
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


def media_kinds(dataset_config: dict[str, Any]) -> list[str]:
    kinds = dataset_config.get("media", list(MEDIA_KINDS))
    return [kinds] if isinstance(kinds, str) else kinds


def count_dataset_samples(dataset_config: dict[str, Any]) -> int:
    """Files only: the dry run does not decode, so split long clips count once."""
    sample_count = count_media_files(str(dataset_config.get("path", "")), media_kinds(dataset_config))
    repeats = int(dataset_config.get("repeats", 1) or 1)
    return max(sample_count, 1) * max(repeats, 1)


def precache(
    config: dict[str, Any],
    samples: list[H3Sample],
    flags: tuple[bool, bool],
    device: torch.device,
    dtype: torch.dtype,
    say: Callable[[str], None],
    show: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Fill ``samples`` (as it goes); returns (caption embeddings, the dataset's composition). ``flags``: (train
    audio, guidance loss)."""
    model_config = config.get("model", {})
    dataset_config = config["dataset"]
    train_audio, _ = flags
    say("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===")
    logger.info("Starting dataset pre-caching...")

    kinds = media_kinds(dataset_config)
    if not train_audio:
        # an audio-only item has nothing but its audio to train
        kinds = [kind for kind in kinds if kind != "audio"]
    sources = scan_media_sources(
        dataset_config["path"],
        kinds=kinds,
        trigger_token=dataset_config.get("trigger", None),
        default_caption=dataset_config.get("default_caption", ""),
    )
    if not sources:
        raise ValueError(f"No training media ({', '.join(kinds)}) found under {dataset_config['path']}")
    logger.info("Dataset: %s", summarize_kinds(sources))
    settings = MediaSettings.from_config(dataset_config, train_audio)

    pending_audio: PendingAudio = []  # audio waits for the audio encoder
    audio_sources = [s for s in sources if s.kind == "audio"]
    visual_sources = [s for s in sources if s.kind != "audio"]
    if visual_sources:  # 1. Video VAE: stills and clip frames
        cache_visual(visual_sources, model_config["vae_path"], settings, (samples, pending_audio), device, say, show)
    if pending_audio or audio_sources:  # 2. Audio VAE: clip soundtracks and audio-only recordings
        cache_audio(model_config, (pending_audio, audio_sources), settings, samples, device, say, show)
    if not samples:
        raise ValueError("Every dataset item was skipped; see the warnings above.")

    text_cache = _cache_prompts(config, samples, flags, (device, dtype), (say, show))
    counts = {
        "images": sum(1 for s in samples if s.kind == "image"),
        "clips": sum(1 for s in samples if s.kind == "clip"),
        "clips_with_audio": sum(1 for s in samples if s.kind == "clip" and s.audio is not None),
        "audio_items": sum(1 for s in samples if s.kind == "audio"),
    }
    logger.info(
        "Pre-cached %d samples (%s) and %d prompt embeddings. Unloaded feature extractors from VRAM.",
        len(samples),
        ", ".join(f"{k}={v}" for k, v in counts.items()),
        len(text_cache),
    )
    return text_cache, counts


def _cache_prompts(
    config: dict[str, Any],
    samples: list[H3Sample],
    flags: tuple[bool, bool],
    on: tuple[torch.device, torch.dtype],
    out: tuple[Callable[[str], None], bool],
) -> dict[str, torch.Tensor]:
    """3. Text encoder: every unique caption, plus the empty prompt when a pass needs it."""
    model_config, train_config = config.get("model", {}), config.get("train", {})
    (device, dtype), (say, show) = on, out
    text_encoder_path = model_config["text_encoder_path"]
    say(f"Loading Text Encoder from {text_encoder_path} for pre-caching...")
    text_encoder = load_minimax_h3_text_encoder(
        text_encoder_path, tokenizer_path=model_config.get("tokenizer_path"), device=device, dtype=dtype
    )
    prompts = {sample.caption for sample in samples}
    if flags[1]:
        prompts.add(str(train_config.get("unconditional_prompt", "")))
    if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
        prompts.add("")
    say(f"Caching text embeddings for {len(prompts)} unique prompts...")
    text_cache: dict[str, torch.Tensor] = {}
    for caption in tqdm(sorted(prompts), desc="Caching Prompts", disable=not show):
        with torch.no_grad():
            text_cache[caption] = text_encoder.encode_prompts([caption], device=device, dtype=dtype).detach().cpu()
    del text_encoder
    free_memory()
    say("Text embeddings pre-cached and Text Encoder unloaded from VRAM.")
    return text_cache
