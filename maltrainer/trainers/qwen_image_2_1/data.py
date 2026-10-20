"""Qwen-Image 2.1's training data: the dataset and its loader, and the latents and caption embeddings, cached
once before the transformer loads (the VAE and the text encoder are unloaded after). Console lines go through
``say`` (the trainer prints them; the library logs)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from maltrainer.data.qwen_dataset import QwenImage21Dataset
from maltrainer.models.qwen_image_2_1.text_encoder import load_qwen_image_2_1_text_encoder
from maltrainer.models.qwen_image_2_1.vae import load_qwen_image_2_1_vae
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


def make_dataset(dataset_config: dict[str, Any], repeats: Any) -> QwenImage21Dataset:
    return QwenImage21Dataset(
        data_root=dataset_config["path"],
        resolution=dataset_config.get("resolution", 512),
        trigger_token=dataset_config.get("trigger"),
        default_caption=dataset_config.get("default_caption", ""),
        repeats=repeats,
    )


def make_loader(config: dict[str, Any]) -> tuple[QwenImage21Dataset, DataLoader[Any]]:
    """The training dataset (with ``dataset.repeats``) and its shuffling loader."""
    dataset_config = config["dataset"]
    dataset = make_dataset(dataset_config, dataset_config.get("repeats", 1))
    loader = DataLoader(
        dataset,
        batch_size=config["train"].get("batch_size", 1),
        shuffle=True,
        num_workers=dataset_config.get("num_workers", 0),
    )
    logger.info("Training dataset configured with %d samples", len(dataset))
    return dataset, loader


def precache(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype, show: bool, say: Callable[[str], None]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(latent per image path, embedding per caption), every image and caption of the dataset encoded once."""
    model_config = config.get("model", {})
    vae_path = model_config["vae_path"]
    text_encoder_path = model_config["text_encoder_path"]

    say("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===")
    logger.info("Starting dataset pre-caching for Qwen-Image-2.1...")
    temp_dataset = make_dataset(config["dataset"], 1)

    # 1. Load VAE & Encode All Unique Images
    say(f"Loading Qwen 2.1 VAE from {vae_path} for pre-caching...")
    vae = load_qwen_image_2_1_vae(vae_path, device=device, dtype=dtype, include_decoder=False)
    latent_cache: dict[str, torch.Tensor] = {}
    say("Caching VAE latents for images...")
    for item in tqdm(
        DataLoader(temp_dataset, batch_size=1, shuffle=False), desc="Caching VAE Latents", disable=not show
    ):
        img_path = item["image_path"][0]
        if img_path not in latent_cache:
            pixel_values = item["pixel_values"].to(device, dtype=dtype)
            with torch.no_grad():
                latent_cache[img_path] = vae.encode(pixel_values, deterministic=True).detach().cpu()
    del vae
    free_memory()
    say("Qwen 2.1 VAE latents pre-cached and VAE unloaded from VRAM.")

    # 2. Load Text Encoder & Encode All Unique Captions
    say(f"Loading Qwen3-VL Text Encoder from {text_encoder_path} for pre-caching...")
    text_encoder = load_qwen_image_2_1_text_encoder(
        text_encoder_path, asset_dir=model_config.get("tokenizer_path"), device=device
    )
    text_cache: dict[str, torch.Tensor] = {}
    unique_captions = list(set(temp_dataset.captions))
    say(f"Caching text embeddings for {len(unique_captions)} unique prompts...")
    for caption in tqdm(unique_captions, desc="Caching Prompts", disable=not show):
        if caption not in text_cache:
            with torch.no_grad():
                text_cache[caption] = text_encoder.encode_prompts([caption], device=device).detach().cpu()

    train_config = config.get("train", {})
    if train_config.get("do_guidance_loss", train_config.get("do_differential_guidance", False)):
        logger.warning("train.do_guidance_loss is not implemented for Qwen-Image 2.1 and is ignored")

    del text_encoder
    free_memory()
    say("Text embeddings pre-cached and Text Encoder unloaded from VRAM.")
    logger.info(
        "Pre-cached %d latents and %d prompt embeddings. Unloaded feature extractors from VRAM.",
        len(latent_cache),
        len(text_cache),
    )
    return latent_cache, text_cache
