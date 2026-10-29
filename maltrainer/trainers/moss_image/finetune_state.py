"""The MOSS-Image finetune's resume state: everything train.resume_from needs (weights, optimizer, prior, EMA, step,
epoch and RNG states). The dataset fingerprint is the shared finetune data's
(maltrainer/trainers/finetune/dataset.py)."""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from maltrainer.models.moss_image.finetune import save_full_model
from maltrainer.trainers.moss_image.finetune_host import (
    STATE_EMA,
    STATE_MODEL,
    STATE_OPTIMIZER,
    STATE_PRIOR,
    STATE_TRAINER,
)
from maltrainer.trainers.moss_image.finetune_prior import FinetunePrior
from maltrainer.utils.ema import OffloadedParameterEMA
from maltrainer.utils.fsutil import replace_directory, torch_save_closed

logger = logging.getLogger(__name__)


class FinetuneState(FinetunePrior):
    epoch: int
    position: int
    epoch_rng: tuple[Any, ...] | None
    ema: OffloadedParameterEMA | None

    def _save_state(self, state_dir: Path, weights: Path | None = None) -> None:
        """Everything train.resume_from needs, written aside and swapped in whole.

        ``weights``: a checkpoint holding exactly the raw weights. The state then hard-links it
        instead of writing the same 10 GB again; the file stays valid if the checkpoint is
        deleted or replaced later, since saves always write a new file.
        """
        started = time.time()
        tmp = state_dir.with_name(state_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        linked = False
        if weights is not None:
            try:
                os.link(weights, tmp / STATE_MODEL)
                linked = True
            except OSError as e:  # another volume, or a filesystem without hard links
                logger.info("Could not hard-link the state weights (%s); writing a copy", e)
        if not linked:
            save_full_model(
                self.transformer,
                self.layout_path,
                tmp / STATE_MODEL,
                metadata={"maltrainer_state_step": str(self.global_step)},
            )
        torch_save_closed(self.optimizer.state_dict(), tmp / STATE_OPTIMIZER)
        if self.prior is not None and self.prior_path is not None:
            try:
                os.link(self.prior_path, tmp / STATE_PRIOR)
            except OSError:
                shutil.copyfile(self.prior_path, tmp / STATE_PRIOR)
        ema_counters = None
        if self.ema is not None:
            tensors, ema_counters = self.ema.state()
            save_file(tensors, str(tmp / STATE_EMA))
        torch_save_closed(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "position": self.position,
                "epoch_rng": self.epoch_rng,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "ema_counters": ema_counters,
                "group_names": [g["name"] for g in self.optimizer.param_groups],
                "param_names": [name for name, _ in self.named_trainable],
                "dataset_fingerprint": self._dataset_fingerprint(),
                "dataset_samples": self._epoch_size(),
            },
            tmp / STATE_TRAINER,
        )
        replace_directory(tmp, state_dir)
        logger.info(
            "Saved resume state (step %d%s) to %s (%.0fs)",
            self.global_step,
            ", weights hard-linked to the checkpoint" if linked else "",
            state_dir,
            time.time() - started,
        )
