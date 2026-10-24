"""Fibo 2.0 preview keys: ``w`` / ``m`` (any keys the config names) render one prompt, ``g`` renders them all.

    sample:
      keys: {w: "photo of a woman", m: "photo of a man"}   # key -> prompt
      steps: 30            # BRIA's pipeline default
      guidance: 5.0
      resolution: 768      # area target, a square; default dataset.resolution
      seed: 42             # the same noise at every press, so images compare across steps
      ema: true            # render with the EMA weights, as the checkpoints save them

The prompts' text features are made while the text encoder is loaded for the dataset, so a press never needs it; the
VAE decoder loads for the press and goes again. Training waits, a press never touches the training RNG (the noise has
its own generator), and the images go to ``<output.path>/samples/step<N>_<key>.png``.
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

from maltrainer.models.flux3.vae import load_flux2_vae_decoder
from maltrainer.trainers.fibo2.sampling import DEFAULT_GUIDANCE, DEFAULT_STEPS, sample_latents
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)

RESERVED = "sqpegh"  # the command keys of KeyCommands; g is taken by the "all prompts" key below
ALL = "g"


def preview_prompts(config: dict[str, Any]) -> dict[str, str]:
    """The ``sample.keys`` of the config: single-character key -> prompt."""
    keys = (config.get("sample") or {}).get("keys") or {}
    bad = [key for key in keys if len(str(key)) != 1 or str(key) in RESERVED]
    if bad:
        raise ValueError(f"sample.keys must be single characters other than {'/'.join(RESERVED)}, got {bad}")
    return {str(key): str(prompt) for key, prompt in keys.items()}


class FiboPreviews:
    """Mixin of the Fibo 2.0 trainer: the preview keys and the rendering behind them."""

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
        """The captions to encode for previews: the prompts and the empty one the guidance is measured against."""
        prompts = set(preview_prompts(self.config).values())
        return prompts | {""} if prompts else set()

    def preview_keys(self) -> dict[str, tuple[str, Callable[[], None]]]:
        prompts = preview_prompts(self.config)
        keys: dict[str, tuple[str, Callable[[], None]]] = {
            key: (f'render "{prompt}"', partial(self.render_previews, [key])) for key, prompt in prompts.items()
        }
        if prompts:
            keys[ALL] = ("render every preview prompt", partial(self.render_previews, list(prompts)))
        return keys

    def render_previews(self, selected: list[str]) -> None:
        """Render the prompts under ``selected`` at the current step; a failure is the caller's to log."""
        if not self.is_main_process:
            return
        sample = self.config.get("sample") or {}
        prompts = preview_prompts(self.config)
        resolution = int(sample.get("resolution", self.config["dataset"].get("resolution", 1024)))
        grid = (resolution // 16, resolution // 16)
        folder = Path(self.config["output"]["path"]) / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        was_training = self.lora_network.training
        self.lora_network.eval()
        weights = self.ema.applied() if self.ema is not None and sample.get("ema", True) else contextlib.nullcontext()
        try:
            with weights, torch.no_grad():
                latents = {
                    key: sample_latents(
                        self.transformer,
                        self._features(prompts[key]),
                        self._features(""),
                        grid,
                        int(sample.get("steps", DEFAULT_STEPS)),
                        float(sample.get("guidance", DEFAULT_GUIDANCE)),
                        int(sample.get("seed", 42)),
                        self.device,
                        self.dtype,
                    ).cpu()
                    for key in selected
                }
        finally:
            self.lora_network.train(was_training)
        for key, image in self._decode(latents).items():
            path = folder / f"step{self.current_step:06d}_{key}.png"
            image.save(path)
            logger.info('Preview [%s] "%s" -> %s', key, prompts[key], path)

    def _features(self, prompt: str) -> torch.Tensor:
        return self.text_cache[prompt].to(self.device, self.dtype).unsqueeze(0)

    def _decode(self, latents: dict[str, torch.Tensor]) -> dict[str, Image.Image]:
        decoder = load_flux2_vae_decoder(self.config["model"]["vae_path"], self.device)
        try:
            out = {}
            for key, latent in latents.items():
                pixels = decoder.decode(latent.to(self.device)).squeeze(0).mul(255).round().byte()
                out[key] = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
            return out
        finally:
            del decoder
            free_memory()
