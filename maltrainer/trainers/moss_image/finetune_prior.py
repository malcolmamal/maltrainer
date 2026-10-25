"""The MOSS-Image finetune's gallery (key g), its preview rows, and its part of prior preservation (the shared start
is maltrainer/trainers/finetune/prior_setup.py): the size-free training sigma and the base checkpoint's weights."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from maltrainer.finetune.prior import weights_from_file
from maltrainer.trainers.finetune.prior_setup import FinetunePriorSetup
from maltrainer.trainers.moss_image.finetune_data import FinetuneData
from maltrainer.trainers.moss_image.finetune_host import STATE_PRIOR
from maltrainer.training.session import free_memory

logger = logging.getLogger(__name__)


class FinetunePrior(FinetunePriorSetup, FinetuneData):
    prior_state_name = STATE_PRIOR
    resume_dir: Path | None
    layout_path: Path
    named_trainable: list[tuple[str, torch.nn.Parameter]]
    global_step: int
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

    def _prior_sigma(self, x0: torch.Tensor) -> torch.Tensor:  # noqa: ARG002 - MOSS-Image's sigma is size-free
        return self._sample_sigma()

    def _original_weights(self) -> contextlib.AbstractContextManager[Any]:
        """The base checkpoint's values in the trained parameters (frozen ones never changed)."""
        return weights_from_file(self.named_trainable, self.layout_path)

    def _original_file(self) -> Path:
        return self.layout_path
