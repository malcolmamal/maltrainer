"""The Krea 2 finetune's saves: the full bf16 model (after waiting for disk space), and the resume state beside it
(its weights hard-linked to the checkpoint)."""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path

import torch

from maltrainer.models.krea2.finetune import save_krea2_model
from maltrainer.trainers.krea2.finetune_host import STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER, FinetuneHost
from maltrainer.utils.fsutil import replace_directory
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


def save_checkpoint(host: FinetuneHost, step: int | str) -> bool:
    """The full model and, with train.save_state, the resume state (its weights hard-linked to
    the checkpoint). A save that does not fit is skipped (the final one waits for space)."""
    output_config = host.config.get("output", {})
    output_dir = Path(output_config["path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = host._name()
    save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
    if not _wait_for_space(output_dir, host._model_bytes() + DISK_MARGIN, step):
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
    if bool(host.config["train"].get("save_state", True)):
        try:
            host._save_state(output_dir / f"{name}_state", save_path)
        except Exception:  # logged with its traceback; the run goes on
            shutil.rmtree(output_dir / f"{name}_state.tmp", ignore_errors=True)
            logger.exception("Writing the resume state for step %s failed; the previous state is kept.", step)
    return True


def save_state(host: FinetuneHost, state_dir: Path, weights: Path) -> None:
    """Everything train.resume_from needs, written aside and swapped in whole."""
    started = time.time()
    tmp = state_dir.with_name(state_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        os.link(weights, tmp / STATE_MODEL)
    except OSError as e:  # another volume, or no hard links
        logger.info("Could not hard-link the state weights (%s); copying", e)
        shutil.copyfile(weights, tmp / STATE_MODEL)
    torch.save(
        {
            "rotation": host.rotator.state_dict(),
            "always": host.always_optimizer.state_dict() if host.always_optimizer is not None else None,
        },
        tmp / STATE_OPTIMIZER,
    )
    torch.save(
        {
            "global_step": host.global_step,
            "epoch": host.epoch,
            "position": host.position,
            "epoch_rng": host.epoch_rng,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "dataset_fingerprint": host._dataset_fingerprint(),
            "dataset_samples": len(host._epoch_order()),
        },
        tmp / STATE_TRAINER,
    )
    replace_directory(tmp, state_dir)
    logger.info("Saved resume state (step %d) to %s (%.0fs)", host.global_step, state_dir, time.time() - started)
