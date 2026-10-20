"""The MOSS-Image finetune's gallery (key g), its preview rows, and prior preservation: the original model's
predictions recorded (or, on resume, reloaded) as the target for the prior term."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from PIL import Image

from maltrainer.finetune.prior import weights_from_file
from maltrainer.trainers.moss_image.finetune_data import FinetuneData
from maltrainer.trainers.moss_image.finetune_host import STATE_PRIOR
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


class FinetunePrior(FinetuneData):
    resume_dir: Path | None
    layout_path: Path
    named_trainable: list[tuple[str, torch.nn.Parameter]]
    global_step: int
    prior_path: Path | None
    transformer: Any

    if TYPE_CHECKING:  # defined further along the finetune's MRO (the checkpoint mixin, the LoRA trainer)

        def _name(self) -> str: ...
        def _sample_sigma(self) -> torch.Tensor: ...

    def _run_gallery(self) -> None:
        """Key g: render the gallery with the weights checkpoints save (the EMA average if on), at
        the preview settings (size, steps, CFG, negative)."""
        ema = self.ema is not None
        p = self.previews

        def render_one(prompt: str, seed: int, label: str) -> Image.Image:
            p.current = f"step {self.global_step}, {label}"
            return self._render_preview(prompt, seed)

        was_training = self.transformer.training
        self.transformer.eval()
        free_memory()
        try:
            self.gallery.run(
                self.global_step,
                render_one,
                self.ema.applied if self.ema is not None else contextlib.nullcontext,
                ready=self._preview_ready,
                label="EMA" if ema else "raw weights",
            )
        finally:
            self.transformer.train(was_training)
            torch.cuda.empty_cache()

    def _preview_variants(self) -> list[tuple[str, Callable[[], contextlib.AbstractContextManager[Any]]]]:
        """With EMA on, two rows: the raw weights (what is training now) and the average (what
        checkpoints save). The noise of batch-1 steps shows in the first and averages out in the
        second."""
        if self.ema is None:
            return [("", contextlib.nullcontext)]
        return [("raw", contextlib.nullcontext), ("ema", self.ema.applied)]

    def _predict(self, noisy: torch.Tensor, sigma: torch.Tensor, caption: str) -> torch.Tensor:
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        return cast(torch.Tensor, self.transformer(noisy.to(self.dtype), sigma, context))

    def _setup_prior(self) -> None:
        """Record (or, on resume, reload) the original model's predictions for the prior term."""
        if self.prior is None:
            return
        captions = self._prior_captions()
        if not captions:
            raise ValueError(
                f"prior.weight > 0, but no caption names a trigger of prior.classes {sorted(self.prior_classes)}"
            )
        if self.resume_dir is not None:
            path = self.resume_dir / STATE_PRIOR
            if not path.exists():
                raise FileNotFoundError(
                    f"prior.weight > 0, but {path} is missing: the resumed run had no prior, and the "
                    "original model it needs is gone. Set prior.weight: 0 to resume without one."
                )
            self.prior.load(path)
            missing = {k: c for k, c in captions.items() if self._rel(k) not in self.prior}
            if missing:
                # images added since the prior was recorded (e.g. new people): record theirs from the
                # original model too, by putting the base checkpoint's values back for the recording
                logger.info(
                    "Prior: %d image(s) added since it was recorded; recording theirs from the original "
                    "weights in %s (~%.0f min)",
                    len(missing),
                    self.layout_path.name,
                    len(missing) * self.prior.samples_per_image / 6 / 60,
                )
                was_training = self.transformer.training
                self.transformer.eval()
                with weights_from_file(self.named_trainable, self.layout_path):
                    self.prior.build(
                        ((self._rel(key), self.latent_cache[key], caption) for key, caption in missing.items()),
                        self._sample_sigma,
                        self._predict,
                        self.device,
                    )
                self.transformer.train(was_training)
                torch.cuda.empty_cache()
        else:
            self.prior.build(
                ((self._rel(key), self.latent_cache[key], caption) for key, caption in captions.items()),
                self._sample_sigma,
                self._predict,
                self.device,
            )
        self.prior_path = Path(self.config["output"]["path"]) / f"{self._name()}_prior.safetensors"
        self.prior_path.parent.mkdir(parents=True, exist_ok=True)
        self.prior.save(self.prior_path)
        self.run_metadata.update(
            {
                "prior_weight": self.prior.weight,
                "prior_samples_per_image": self.prior.samples_per_image,
                "prior_every": self.prior.every,
                "prior_classes": self.prior_classes,
                "prior_images": len(self.prior),
            }
        )
        logger.info(
            "Prior preservation: weight %g on %d images, every %d step(s), classes %s",
            self.prior.weight,
            len(self.prior),
            self.prior.every,
            self.prior_classes,
        )
