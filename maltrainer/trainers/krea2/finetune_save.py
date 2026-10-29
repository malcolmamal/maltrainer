"""The Krea 2 finetune's saves: the full bf16 model (after waiting for disk space), and the resume state beside it
(its weights hard-linked to the checkpoint, and the prior's recorded predictions to <name>_prior.safetensors)."""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from maltrainer.models.krea2.finetune import save_krea2_model
from maltrainer.trainers.krea2.finetune_host import (
    STATE_MODEL,
    STATE_OPTIMIZER,
    STATE_PRIOR,
    STATE_TRAINER,
    FinetuneHost,
)
from maltrainer.utils.fsutil import remove_unfinished, replace_directory, torch_save_closed
from maltrainer.utils.lora_metadata import build_metadata

logger = logging.getLogger(__name__)

DISK_MARGIN = 1 << 30


def _wait_for_space(output_dir: Path, need: int, step: int | str) -> bool:
    """False when a numbered save does not fit (it is skipped, training continues); the final one waits."""
    while shutil.disk_usage(output_dir).free < need:
        free = shutil.disk_usage(output_dir).free
        if step != "final":
            logger.error(
                "Not saving step %s: it needs %.1f GB and %s has %.1f GB free. Training continues.",
                step,
                need / 1e9,
                output_dir,
                free / 1e9,
            )
            return False
        logger.error(
            "The final model needs %.1f GB and %s has %.1f GB free: free space there; checking again every 30 s.",
            need / 1e9,
            output_dir,
            free / 1e9,
        )
        time.sleep(30)
    return True


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def state_bytes(host: FinetuneHost) -> int:
    """What the resume state writes beside its hard-linked model: every window's optimizer state (the active
    one read where it lives, not copied) and the text fusion's."""
    rotator = host.rotator
    states: list[Any] = [rotator.optimizer_states]
    if rotator.current is not None:
        states.append(rotator.optimizer.state_dict())
    if host.always_optimizer is not None:
        states.append(host.always_optimizer.state_dict())
    return sum(_tensor_bytes(state) for state in states)


def save_checkpoint(host: FinetuneHost, step: int | str) -> bool:
    """The full model and, with train.save_state, the resume state (its weights hard-linked to
    the checkpoint). A save that does not fit is skipped (the final one waits for space)."""
    output_config = host.config.get("output", {})
    output_dir = Path(output_config["path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = host._name()
    save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
    with_state = bool(host.config["train"].get("save_state", True))
    need = host._model_bytes() + (state_bytes(host) if with_state else 0) + DISK_MARGIN
    if not _wait_for_space(output_dir, need, step):
        return False
    runtime = dict(
        host.run_metadata,
        saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        global_step=host.global_step,
        save_dtype="bfloat16",
    )
    metadata = build_metadata(
        host.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True))
    )
    started = time.time()
    try:
        save_krea2_model(host.transformer, host.rotator, save_path, host.key_prefix, metadata)
    except Exception as error:
        Path(str(save_path) + ".tmp").unlink(missing_ok=True)
        logger.error("Saving step %s failed (%s); the partial file is removed and training continues.", step, error)
        if step == "final":
            raise
        return False
    host.last_saved_path = str(save_path)
    logger.info("Saved finetuned model to %s (%.0fs)", save_path, time.time() - started)
    if with_state:
        try:
            host._save_state(output_dir / f"{name}_state", save_path)
        except Exception:  # logged with its traceback; the run goes on
            logger.exception("Writing the resume state for step %s failed; the previous state is kept.", step)
            remove_unfinished(output_dir / f"{name}_state.tmp", "resume state")
    return True


def _link(source: Path, target: Path, what: str) -> None:
    """A hard link to ``source``, or a copy where there can be none (another volume, a filesystem without them)."""
    try:
        os.link(source, target)
    except OSError as e:
        logger.info("Could not hard-link the %s (%s); copying", what, e)
        shutil.copyfile(source, target)


def save_state(host: FinetuneHost, state_dir: Path, weights: Path) -> None:
    """Everything train.resume_from needs, written aside and swapped in whole."""
    started = time.time()
    tmp = state_dir.with_name(state_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _link(weights, tmp / STATE_MODEL, "state weights")
    if host.prior is not None and host.prior_path is not None:
        _link(host.prior_path, tmp / STATE_PRIOR, "prior")
    torch_save_closed(
        {
            "rotation": host.rotator.state_dict(),
            "always": host.always_optimizer.state_dict() if host.always_optimizer is not None else None,
        },
        tmp / STATE_OPTIMIZER,
    )
    torch_save_closed(
        {
            "global_step": host.global_step,
            "epoch": host.epoch,
            "position": host.position,
            "epoch_rng": host.epoch_rng,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "dataset_fingerprint": host._dataset_fingerprint(),
            "dataset_samples": host._epoch_size(),  # counted, not drawn: a save leaves Python's random alone
        },
        tmp / STATE_TRAINER,
    )
    replace_directory(tmp, state_dir)
    logger.info("Saved resume state (step %d) to %s (%.0fs)", host.global_step, state_dir, time.time() - started)
