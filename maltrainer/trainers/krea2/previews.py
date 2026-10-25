"""Krea 2 previews while finetuning: MOSS-Image's preview plumbing (maltrainer/trainers/moss_image/previews.py: the
sample: section reloaded when the config file is saved, prompts added mid-run encoded on the CPU, the menu on key e)
with Krea 2's sampler (maltrainer/models/krea2/sampling.py) and Wan 2.1 decoder, and the gallery on key g.

The decoder loads on the first render and stays (~0.2 GB in bf16). A render runs the DiT as it trains: the frozen
windows in fp8, the active one and the text fusion in bf16. Prompts added mid-run are encoded by Qwen3-VL-4B on the
CPU (~8 GB of RAM while it is loaded), never on the GPU beside training.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from maltrainer.models.krea2.sampling import sample as krea2_sample
from maltrainer.models.krea2.sampling import size_problem
from maltrainer.models.krea2.text_encoder import load_krea2_text_encoder
from maltrainer.models.krea2.vae_decoder import load_krea2_vae_decoder
from maltrainer.trainers.moss_image.previews import PreviewHooks
from maltrainer.training.session import free_memory
from maltrainer.utils.gallery import Gallery

logger = logging.getLogger(__name__)


class Krea2Previews(PreviewHooks):
    """Mixed into the Krea 2 finetune, which sets these."""

    transformer: Any
    global_step: int
    gallery: Gallery

    @torch.no_grad()
    def _render_preview(self, prompt: str, seed: int) -> Image.Image:
        """One preview with the weights as they are now: the Krea 2 sampler, then the Wan 2.1 decoder."""
        p = self.previews
        if self._preview_decoder is None:
            self._preview_decoder = load_krea2_vae_decoder(self.config["model"]["vae_path"], self.device, self.dtype)
        context = self.text_cache[prompt].to(self.device, self.dtype)
        negative = self.text_cache[p.negative].to(self.device, self.dtype) if p.cfg != 1.0 else None
        with tqdm(total=p.steps, desc=f"Sampling ({p.current})", leave=False, disable=not self.is_main_process) as bar:

            def on_step(done: int, total: int) -> None:
                bar.total = total
                bar.update(done - bar.n)

            latents = krea2_sample(
                self.lora_network,
                context,
                p.height,
                p.width,
                steps=p.steps,
                seed=seed,
                cfg=p.cfg,
                negative=negative,
                channels=self.transformer.config.channels,
                device=self.device,
                on_step=on_step,
            )
        image = self._preview_decoder.decode(latents)[0]
        return Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))

    def _check_preview_size(self) -> None:
        """At construction: a size the sampler cannot render stops the run before any caching, not at step 0."""
        problem = size_problem(self.previews.width, self.previews.height)
        if problem:
            raise ValueError(problem)

    def _watch_config(self) -> None:
        """The sample: section reloaded when the file is saved; a size the sampler cannot render keeps the last good
        one (logged), so a typo mid-run never stops the training."""
        before = (self.previews.width, self.previews.height)
        super()._watch_config()
        problem = size_problem(self.previews.width, self.previews.height)
        if problem:
            logger.error("[sample] %s; keeping %dx%d", problem, *before)
            self.previews.width, self.previews.height = before

    def _load_cpu_encoder(self) -> Any:
        model_config = self.config["model"]
        return load_krea2_text_encoder(
            model_config["text_encoder_path"], torch.device("cpu"), self.dtype, model_config.get("tokenizer_path")
        )

    def _run_gallery(self) -> None:
        """Key g: the gallery (maltrainer/utils/gallery.py) with the weights in training (a Krea 2 finetune has no
        EMA), at the preview settings."""
        p = self.previews

        def render_one(prompt: str, seed: int, label: str) -> Image.Image:
            p.current = f"step {self.global_step}, {label}"
            return self._render_preview(prompt, seed)

        was_training = self.transformer.training
        self.transformer.eval()
        free_memory()
        try:
            self.gallery.run(self.global_step, render_one, contextlib.nullcontext, ready=self._preview_ready)
        finally:
            self.transformer.train(was_training)
            torch.cuda.empty_cache()
