"""MOSS-Image previews while training: rendering with the weights in training, the live reload of the config's
sample: section, encoding prompts added mid-run, and the preview menu (key e). The finetune inherits all of it."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from maltrainer.models.flux3.vae import load_flux2_vae_decoder
from maltrainer.models.moss_image.sampling import sample as moss_sample
from maltrainer.models.moss_image.text_encoder import load_moss_image_text_encoder
from maltrainer.training.session import free_memory
from maltrainer.utils.config import load_config
from maltrainer.utils.console_keys import read_key
from maltrainer.utils.ema import OffloadedParameterEMA, ParameterEMA
from maltrainer.utils.preview_menu import run_menu
from maltrainer.utils.previews import Previews

logger = logging.getLogger(__name__)


class PreviewHooks:
    """Mixed into the MOSS-Image LoRA trainer, which sets these."""

    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    text_cache: dict[str, torch.Tensor]
    lora_network: Any
    is_main_process: bool
    ema: ParameterEMA | OffloadedParameterEMA | None
    previews: Previews
    _preview_decoder: Any
    config_path: Path | None  # the config file, watched for edits to its sample: section (set by main)
    _config_mtime: float | None
    _cpu_encoder: Any
    _encoding: threading.Thread | None

    def _init_previews(self, config: dict[str, Any]) -> None:
        # sample: previews of fixed prompts every N steps, rendered with the weights in training
        self.previews = Previews(config.get("sample"), Path(config.get("output", {}).get("path", "./output")))
        self._preview_decoder = None
        self.config_path = None
        self._config_mtime = None
        self._cpu_encoder = None
        self._encoding = None

    def _extra_prompts(self) -> list[str]:
        """Captions to encode beside the samples' own: the preview prompts (and the finetune's
        prior-preservation ones)."""
        return self.previews.captions

    @torch.no_grad()
    def _render_preview(self, prompt: str, seed: int) -> Image.Image:
        """One preview with the weights as they are now: the reference sampler, then the VAE decoder."""
        p = self.previews
        if self._preview_decoder is None:
            self._preview_decoder = load_flux2_vae_decoder(self.config["model"]["vae_path"], self.device)
        context = self.text_cache[prompt].to(self.device, self.dtype)
        negative = self.text_cache[p.negative].to(self.device, self.dtype) if p.cfg != 1.0 else None
        with tqdm(total=p.steps, desc=f"Sampling ({p.current})", leave=False, disable=not self.is_main_process) as bar:

            def on_step(done: int, total: int) -> None:
                bar.total = total
                bar.update(done - bar.n)

            latents = moss_sample(
                self.lora_network,
                context,
                p.height,
                p.width,
                steps=p.steps,
                seed=seed,
                cfg=p.cfg,
                negative=negative,
                device=self.device,
                on_step=on_step,
            )
        image = self._preview_decoder.decode(latents)[0]
        return Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))

    def _watch_config(self) -> None:
        """At a step boundary: if the config file was saved since the last look, apply its sample:
        section (interval, prompts, settings) without stopping. Other sections are only read at
        the start; edits to them are reported and ignored until the next start."""
        if self.config_path is None or not self.is_main_process:
            return
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return
        if self._config_mtime is None:
            self._config_mtime = mtime
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        try:
            fresh = load_config(str(self.config_path))
        except (OSError, ValueError) as error:  # e.g. a half-edited YAML file
            logger.warning(
                "[sample] %s could not be read (%s); the previous settings stay", self.config_path.name, error
            )
            return
        ignored = sorted(
            k for k in set(fresh) | set(self.config) if k != "sample" and fresh.get(k) != self.config.get(k)
        )
        if ignored:
            logger.info(
                "[sample] %s changed outside sample: (%s): those apply at the next start",
                self.config_path.name,
                ", ".join(ignored),
            )
        changes = self.previews.update(fresh.get("sample"))
        self.config["sample"] = fresh.get("sample")
        if changes:
            logger.info("[sample] reloaded from %s:\n  %s", self.config_path.name, "\n  ".join(changes))
        self._encode_new_prompts()

    def _encode_new_prompts(self) -> None:
        """Preview prompts not encoded yet, on the CPU in the background (~6 s each): the GPU is
        busy training, and the text encoder (~10 GB) does not fit beside it. The int8 layers run
        their float fallback on the CPU, so these embeddings differ ~2.6% from the GPU path's;
        they serve previews only and never enter the caption cache."""
        missing = [c for c in dict.fromkeys(self.previews.captions) if c not in self.text_cache]
        if not missing or (self._encoding is not None and self._encoding.is_alive()):
            return

        def work() -> None:
            try:
                if self._cpu_encoder is None:
                    model_config = self.config["model"]
                    self._cpu_encoder = load_moss_image_text_encoder(
                        model_config["text_encoder_path"],
                        torch.device("cpu"),
                        self.dtype,
                        model_config.get("tokenizer_path"),
                    )
                for caption in missing:
                    started = time.time()
                    self.text_cache[caption] = self._cpu_encoder.encode(caption, torch.device("cpu"), self.dtype).cpu()
                    logger.info(
                        "[sample] encoded on the CPU in %.0fs: %s", time.time() - started, caption or "(empty negative)"
                    )
            except Exception:  # a background thread: log it with its traceback, do not die silently
                logger.exception("[sample] encoding new prompts failed")

        logger.info("[sample] encoding %d new prompt(s) on the CPU in the background", len(missing))
        self._encoding = threading.Thread(target=work, daemon=True)
        self._encoding.start()

    def _preview_menu(self) -> None:
        """Key e: the preview settings menu. Rows to choose from are the labelled variants (a
        finetune with EMA: raw and ema); prompts added mid-run are encoded first if needed."""
        run_menu(self.previews, self._preview_row_labels(), read_key, ready=self._preview_ready)
        self._encode_new_prompts()

    def _preview_row_labels(self) -> list[str]:
        """The rows the preview settings can switch (the menu's, and the control API's)."""
        return [label for label, _ in self._preview_variants() if label]

    def _preview_ready(self, prompt: str) -> bool:
        p = self.previews
        return prompt in self.text_cache and (p.cfg == 1.0 or p.negative in self.text_cache)

    def _preview_variants(self) -> list[tuple[str, Callable[[], contextlib.AbstractContextManager[Any]]]]:
        """(label, weights context) per row of the preview strip. LoRA checkpoints save the EMA
        average, so the previews show it."""
        if self.ema is not None and hasattr(self.ema, "applied"):
            return [("", self.ema.applied)]
        return [("", contextlib.nullcontext)]

    def _maybe_preview(self, step: int, total: int) -> float:
        """Render the previews if they are due; returns the seconds spent (0 if not)."""
        if not self.is_main_process or not self.previews.due(step, total):
            return 0.0
        was_training = self.lora_network.training
        self.lora_network.eval()
        free_memory()  # hand training's cached-but-unused memory back first, so the render does not stack on it
        try:
            return self.previews.render(step, self._render_preview, self._preview_variants(), ready=self._preview_ready)
        finally:
            self.lora_network.train(was_training)
            torch.cuda.empty_cache()
