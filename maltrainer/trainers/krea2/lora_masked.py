"""Krea 2 LoRA training with masks: ``masks.enabled`` weights each picture's loss by the mask beside it.

Only the LoRA entry point uses this class; the finetune trainer builds on ``Krea2LoRATrainer`` itself and never sees it.
A picture without a mask (or with one that can't be used) trains as normal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from maltrainer.data.masks import is_mask_file, latent_mask, mask_for, mask_problem
from maltrainer.data.minimax_dataset import MediaSource, scan_media_sources
from maltrainer.training.masked_loss import mask_weight, prior_loss, weighted_mse
from train_krea2_lora import Krea2LoRATrainer

logger = logging.getLogger(__name__)
VAE_SCALE = 8  # pixels per latent cell


class MaskedKrea2LoRATrainer(Krea2LoRATrainer):
    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        masks = config.get("masks", {}) or {}
        self.masked = bool(masks.get("enabled", False))
        self.mask_floor = float(masks.get("floor", 0.1))
        self.mask_prior = float(masks.get("prior", 0.0))
        self.mask_cache: dict[str, torch.Tensor] = {}  # sample key -> [1, 1, h, w] on its latent grid

    def _sources(self) -> list[MediaSource]:
        return [source for source in super()._sources() if not is_mask_file(source.path)]

    def _dry_run_samples(self) -> int:
        dataset = self.config["dataset"]
        found = scan_media_sources(dataset.get("path", ""), ["image"])
        repeats = max(int(dataset.get("repeats", 1) or 1), 1)
        return max(sum(not is_mask_file(s.path) for s in found), 1) * repeats

    def precache_dataset(self) -> None:
        super().precache_dataset()
        if self.masked:
            self._cache_masks()

    def _cache_masks(self) -> None:
        for key, _ in self.samples:
            path = mask_for(key)
            if path is None:
                continue
            with Image.open(key) as image:
                size = image.size
            problem = mask_problem(path, size)
            if problem:
                logger.warning("Mask %s %s; %s trains without it.", path.name, problem, Path(key).name)
                continue
            height, width = self.latent_cache[key].shape[-2:]
            self.mask_cache[key] = latent_mask(path, size, (width * VAE_SCALE, height * VAE_SCALE), (height, width))
        if not self.mask_cache:
            raise ValueError(
                "masks.enabled is on but no image has a usable mask (photo-masklabel.png beside photo.png)"
            )
        logger.info(
            "Masked training: %d of %d images have a mask (floor %.2f, prior %.2f)",
            len(self.mask_cache),
            len(self.samples),
            self.mask_floor,
            self.mask_prior,
        )
        self.run_metadata["masked_training"] = {
            "floor": self.mask_floor,
            "prior": self.mask_prior,
            "images_with_mask": len(self.mask_cache),
        }

    def _training_step(
        self, index: int, shift_logged: list[bool] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.mask_cache.get(self.samples[index][0]) if self.masked else None
        if mask is None:
            return super()._training_step(index, shift_logged)
        return self._masked_step(index, mask, self._shift_logged if shift_logged is None else shift_logged)

    def _masked_step(
        self, index: int, mask: torch.Tensor, shift_logged: list[bool]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``Krea2LoRATrainer._training_step`` with the mask weighting the error (and the prior outside it)."""
        key, caption = self._sample(index)
        x0 = self.latent_cache[key].to(self.device, torch.float32)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        sigma, shift = self._sample_sigma((x0.shape[-2] // 2) * (x0.shape[-1] // 2))
        if not shift_logged[0]:
            shift_logged[0] = True
            self.run_metadata["resolved_sigma_shift"] = round(shift, 6)
        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = ((1.0 - s) * x0 + s * noise).to(self.dtype)
        weight = mask_weight(mask.to(self.device), self.mask_floor)
        base = None
        if self.mask_prior > 0:
            with torch.no_grad(), self.accelerator.unwrap_model(self.lora_network).disabled():
                base = self.lora_network(noisy, sigma, context).float()
        pred = self.lora_network(noisy, sigma, context)
        loss = weighted_mse(pred, noise - x0, weight)
        if base is not None:
            loss = loss + prior_loss(pred, base, weight, self.mask_prior)
        return loss * float(self.config["train"].get("loss_weight_strength", 1.0)), noisy, sigma
