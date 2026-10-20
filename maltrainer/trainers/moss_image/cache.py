"""MOSS-Image's caching: VAE latents and caption embeddings, each read from the image folders' disk caches
(``dataset.cache``) where they are, encoded (and stored there) where not; the encoder loads only if needed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.data.disk_cache import FolderCache, cache_key, content_hash, model_identity
from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.data.minimax_dataset import MediaSource
from maltrainer.models.flux3.vae import load_flux2_vae_encoder
from maltrainer.models.moss_image.text_encoder import MAX_SEQUENCE_LENGTH, TEMPLATE, load_moss_image_text_encoder
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


def cache_latents(
    sources: list[MediaSource], config: dict[str, Any], device: torch.device, show: bool
) -> tuple[dict[str, torch.Tensor], int]:
    """(image path -> latent for every source that encodes, how many were encoded rather than read); a failing
    one is skipped with a warning."""
    model_config, dataset_config = config["model"], config["dataset"]
    resolution = int(dataset_config.get("resolution", 512))
    aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()
    use_disk = bool(dataset_config.get("cache", True))  # each image folder keeps a .maltrainer_cache

    vae_id = model_identity(model_config["vae_path"])
    latents: dict[str, torch.Tensor] = {}
    pending = []
    for source in sources:
        folder = FolderCache(source.path.parent)
        key = (
            cache_key("moss_image_latent/v1", content_hash(source.path), resolution, aspect_mode, vae_id)
            if use_disk
            else None
        )
        latent = folder.get("latents", key) if use_disk and key is not None else None
        if latent is None:
            pending.append((source, folder, key))
        else:
            latents[str(source.path)] = latent
    if pending:
        vae = load_flux2_vae_encoder(model_config["vae_path"], device)
        for source, folder, key in tqdm(pending, desc="Caching latents", disable=not show):
            try:
                pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(device)
                latent = vae.encode(pixels)[0].cpu()
                latents[str(source.path)] = latent
                if use_disk:
                    folder.put("latents", key, latent)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        free_memory()
    return latents, len(pending)


def cache_prompts(
    prompt_folders: dict[str, set[Any]], config: dict[str, Any], on: tuple[torch.device, torch.dtype], show: bool
) -> dict[str, torch.Tensor]:
    """Caption -> embedding. A caption's embedding lives in every folder ``prompt_folders`` names for it."""
    model_config = config["model"]
    device, dtype = on
    use_disk = bool(config["dataset"].get("cache", True))
    prompts = sorted(prompt_folders)
    text_id = cache_key(
        "moss_image_text/v1",
        model_identity(model_config["text_encoder_path"]),
        model_config.get("tokenizer_path"),
        TEMPLATE,
        MAX_SEQUENCE_LENGTH,
        dtype,
    )
    text_cache: dict[str, torch.Tensor] = {}
    pending_prompts = []
    for caption in prompts:
        key = cache_key(text_id, caption)
        folders = [FolderCache(folder) for folder in sorted(prompt_folders[caption])]
        embeds = _stored(folders, key) if use_disk else None
        if embeds is None:
            pending_prompts.append((caption, key, folders))
            continue
        text_cache[caption] = embeds
        for folder in folders:  # e.g. a caption now shared with a new folder
            if not folder.has("text", key):
                folder.put("text", key, embeds)
    if pending_prompts:
        encoder = load_moss_image_text_encoder(
            model_config["text_encoder_path"], device, dtype, model_config.get("tokenizer_path")
        )
        for caption, key, folders in tqdm(pending_prompts, desc="Caching prompts", disable=not show):
            text_cache[caption] = encoder.encode(caption, device, dtype).cpu()
            if use_disk:
                for folder in folders:
                    folder.put("text", key, text_cache[caption])
        del encoder
        free_memory()
    logger.info(
        "Prompt embeddings: %d from the disk cache, %d encoded",
        len(prompts) - len(pending_prompts),
        len(pending_prompts),
    )
    return text_cache


def collect_samples(
    sources: list[MediaSource],
    latents: dict[str, torch.Tensor],
    samples: list[tuple[str, str]],
    latent_cache: dict[str, torch.Tensor],
) -> dict[tuple[int, int], int]:
    """Fill ``samples`` and ``latent_cache`` in source order (whatever came from disk, so a resumed run shuffles
    the same samples); returns images per training size (w, h)."""
    sizes: dict[tuple[int, int], int] = {}
    for source in sources:
        latent = latents.get(str(source.path))
        if latent is None:
            continue
        latent_cache[str(source.path)] = latent
        samples.append((str(source.path), source.caption))
        size = (latent.shape[-1] * 16, latent.shape[-2] * 16)
        sizes[size] = sizes.get(size, 0) + 1
    return sizes


def _stored(folders: list[FolderCache], key: str) -> torch.Tensor | None:
    for folder in folders:
        if folder.has("text", key) and (embeds := folder.get("text", key)) is not None:
            return embeds
    return None


def prompt_folders(
    samples: list[tuple[str, str]], dataset_path: str, extra: list[str], dropout: bool
) -> dict[str, set[Any]]:
    """Caption -> the folders whose images use it; the empty dropout prompt and the extra prompts (previews,
    prior captions) in the dataset folder itself."""
    folders: dict[str, set[Any]] = {}
    for key, caption in samples:
        folders.setdefault(caption, set()).add(Path(key).parent)
    if dropout:
        folders.setdefault("", set()).add(Path(dataset_path))
    for caption in extra:
        folders.setdefault(caption, set()).add(Path(dataset_path))
    return folders
