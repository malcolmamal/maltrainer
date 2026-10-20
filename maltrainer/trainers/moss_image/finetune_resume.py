"""The MOSS-Image finetune's start of a run: the EMA, where a saved run stands (and whether it was on this data),
and the training monitor."""

from __future__ import annotations

import logging
from typing import Any

from safetensors.torch import load_file

from maltrainer.models.moss_image.finetune import drift_targets
from maltrainer.trainers.moss_image.finetune_host import STATE_EMA
from maltrainer.utils.ema import OffloadedParameterEMA
from maltrainer.utils.train_monitor import TrainMonitor, WeightDrift

logger = logging.getLogger(__name__)


def accumulation(train_config: dict[str, Any]) -> int:
    return max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)


def start_ema(host: Any, train_config: dict[str, Any]) -> None:
    ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
    if ema_decay > 0:
        host.ema = OffloadedParameterEMA(
            host.named_trainable,
            decay=ema_decay,
            warmup=bool(train_config.get("ema_warmup", True)),
            every=int(train_config.get("ema_every", 1) or 1),
        )
        host.run_metadata.update(ema_decay=ema_decay, ema_every=host.ema.every)
        logger.info(
            "EMA in CPU RAM: decay %.4f every %d step(s) (checkpoints save the average)", ema_decay, host.ema.every
        )


def resume_position(host: Any, resume: dict[str, Any]) -> bool:
    """Where the saved run stands (and its EMA); whether it was on this dataset (else a fresh epoch starts)."""
    host.global_step, host.epoch, host.position = resume["global_step"], resume["epoch"], resume["position"]
    if host.ema is not None and host.resume_dir is not None:
        if not (host.resume_dir / STATE_EMA).exists():
            logger.warning("The resumed run had no EMA; the average starts from the current weights.")
        else:
            host.ema.load_state(load_file(str(host.resume_dir / STATE_EMA)), resume.get("ema_counters", {}))
    # Older states carry no fingerprint; treat them as the same data.
    same_data = resume.get("dataset_fingerprint", host._dataset_fingerprint()) == host._dataset_fingerprint()
    if same_data:
        logger.info(
            "Resumed at step %d (epoch %d, sample %d of its order)", host.global_step, host.epoch + 1, host.position
        )
        return True
    # The old epoch's shuffled order means nothing for a different set: carry on
    # with a fresh epoch. Weights, optimizer, EMA and step count still continue.
    if host.position > 0:
        host.epoch += 1
    host.position = 0
    logger.warning(
        "The dataset changed since the saved state (%s samples then, %d now); resuming at step %d with a fresh epoch.",
        resume.get("dataset_samples", "?"),
        host._epoch_size(),
        host.global_step,
    )
    return False


def build_monitor(host: Any, train_config: dict[str, Any]) -> TrainMonitor:
    """train.monitor_every: a health summary line (loss by noise band, prior, the optimizer's direction signal,
    weight drift, NaN counts, speed, VRAM); maltrainer/utils/train_monitor.py."""
    drift = None
    if bool(train_config.get("monitor_drift", True)):
        tracked = drift_targets(host.named_trainable)
        if tracked:
            drift = WeightDrift(tracked)
            logger.info("Monitor: weight drift of %d tensors (%.0f MB of copies)", len(tracked), drift.size_mb)
    return TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100), drift=drift)
