"""Visage-Image preview keys: ``w`` / ``m`` (any keys the config names) render one prompt, ``g`` renders them all.

    sample:
      keys: {w: "a young woman, smiling", m: "a middle-aged man, grey beard"}   # key -> prompt
      prompts: ["an old woman with white hair"]    # more prompts, rendered by g only (named p1, p2 ...)
      steps: 50            # the release recipe: sde, churn 0.3, cfg 3, shift 3.3
      guidance: 3.0
      solver: sde          # or euler
      churn: 0.3
      shift: 3.3
      resolution: 256      # a square; default dataset.resolution
      seed: 42             # the same noise at every press, so images compare across steps
      ema: true            # render with the EMA weights, as the checkpoints save them

The prompts' text features are made while the text encoder is loaded for the dataset, so a press never needs it; the
VAE loads for the press and goes again. Training waits, a press never touches the training RNG (the noise has its own
generator), and the images go to ``<output.path>/samples/step<N>_<key>.png``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from maltrainer.models.visage.vae import load_visage_vae
from maltrainer.trainers.fibo2.previews import ALL, gallery_prompts, preview_prompts
from maltrainer.trainers.visage.sampling import (
    DEFAULT_CHURN,
    DEFAULT_GUIDANCE,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    sample_latents,
)
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


class VisagePreviews:
    """Mixin of the Visage-Image trainer: the preview keys and the rendering behind them."""

    config: dict[str, Any]
    transformer: Any
    text_cache: dict[str, torch.Tensor]
    device: torch.device
    dtype: torch.dtype
    ema: Any
    is_main_process: bool
    lora_network: Any
    current_step: int = 0

    def preview_texts(self) -> set[str]:
        """The captions to encode for previews: the prompts."""
        return set(gallery_prompts(self.config).values())

    def preview_keys(self) -> dict[str, tuple[str, Callable[[], None]]]:
        keys: dict[str, tuple[str, Callable[[], None]]] = {
            key: (f'render "{prompt}"', partial(self.render_previews, [key]))
            for key, prompt in preview_prompts(self.config).items()
        }
        everything = gallery_prompts(self.config)
        if everything:
            keys[ALL] = (
                f"render every preview prompt ({len(everything)})",
                partial(self.render_previews, list(everything)),
            )
        return keys

    def render_previews(self, selected: list[str]) -> None:
        """Render the prompts under ``selected`` at the current step; a failure is the caller's to log."""
        if not self.is_main_process:
            return
        sample = self.config.get("sample") or {}
        prompts = gallery_prompts(self.config)
        side = int(sample.get("resolution", self.config["dataset"].get("resolution", 256))) // 16
        folder = Path(self.config["output"]["path"]) / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        was_training = self.lora_network.training
        self.lora_network.eval()
        weights = self.ema.applied() if self.ema is not None and sample.get("ema", True) else contextlib.nullcontext()
        try:
            with weights, torch.no_grad():
                latents = {key: self._render_one(prompts[key], key, (side, side), sample) for key in selected}
        finally:
            self.lora_network.train(was_training)
        for key, image in self._decode(latents).items():
            path = folder / f"step{self.current_step:06d}_{key}.png"
            image.save(path)
            logger.info('Preview [%s] "%s" -> %s', key, prompts[key], path)

    def _render_one(self, prompt: str, key: str, grid: tuple[int, int], sample: dict[str, Any]) -> torch.Tensor:
        """One prompt's latents, with a progress bar over the steps."""
        steps = int(sample.get("steps", DEFAULT_STEPS))
        with tqdm(total=steps, desc=f"Sampling ({key})", leave=False) as bar:
            return sample_latents(
                self.transformer,
                self.text_cache[prompt].to(self.device, self.dtype).unsqueeze(0),
                grid,
                int(sample.get("seed", 42)),
                self.device,
                self.dtype,
                steps,
                float(sample.get("guidance", DEFAULT_GUIDANCE)),
                str(sample.get("solver", "sde")),
                float(sample.get("churn", DEFAULT_CHURN)),
                float(sample.get("shift", DEFAULT_SHIFT)),
                lambda done, _total: bar.update(done - bar.n),
            ).cpu()

    def _decode(self, latents: dict[str, torch.Tensor]) -> dict[str, Image.Image]:
        vae = load_visage_vae(self.config["model"]["vae_path"], self.device)
        try:
            out = {}
            for key, latent in latents.items():
                pixels = vae.decode(latent.to(self.device)).squeeze(0).mul(255).round().byte()
                out[key] = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
            return out
        finally:
            del vae
            free_memory()
