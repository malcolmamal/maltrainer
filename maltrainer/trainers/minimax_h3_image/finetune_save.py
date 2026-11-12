"""The H3 Image finetune's saves: the INT8 ConvRot model (after waiting for disk space), and the resume state beside
it: the bf16 value of every trained weight, every window's optimizer state, and the counters, RNG states and
dataset fingerprint that make a resumed run end bit for bit where an uninterrupted one does."""

from __future__ import annotations

import logging
import random
import shutil
import time
from pathlib import Path

import torch

from maltrainer.models.minimax_h3_image.finetune_save import write_int8_model, write_resume_weights
from maltrainer.trainers.krea2.finetune_save import DISK_MARGIN, _tensor_bytes, _wait_for_space
from maltrainer.trainers.minimax_h3_image.finetune_host import (
    STATE_OPTIMIZER,
    STATE_TRAINER,
    STATE_WEIGHTS,
    FinetuneHost,
)
from maltrainer.utils.fsutil import remove_unfinished, replace_directory, torch_save_closed
from maltrainer.utils.lora_metadata import build_metadata

logger = logging.getLogger(__name__)


def state_bytes(host: FinetuneHost) -> int:
    """What the resume state writes: every trained weight in bf16 (an upper bound: all of them) and the optimizers'."""
    rotator = host.rotator
    states: list[object] = [rotator.optimizer_states]
    if rotator.current is not None:
        states.append(rotator.optimizer.state_dict())
    if host.always_optimizer is not None:
        states.append(host.always_optimizer.state_dict())
    weights = 2 * sum(host.rotator.window_parameters(i) for i in range(len(host.rotator.windows)))
    return weights + sum(_tensor_bytes(state) for state in states)


def save_checkpoint(host: FinetuneHost, step: int | str) -> bool:
    """The INT8 model and, with train.save_state, the resume state. A save that does not fit is skipped (the final
    one waits for space)."""
    output_config = host.config.get("output", {})
    output_dir = Path(output_config["path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name, source = host._name(), Path(host.config["model"]["transformer_path"])
    save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
    with_state = bool(host.config["train"].get("save_state", True))
    need = source.stat().st_size + (state_bytes(host) if with_state else 0) + DISK_MARGIN
    if not _wait_for_space(output_dir, need, step):
        return False
    runtime = dict(host.run_metadata, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), global_step=host.global_step)
    metadata = build_metadata(
        host.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True))
    )
    started = time.time()
    try:
        write_int8_model(source, host.rotator, host.always_on, save_path, metadata)
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
            host._save_state(output_dir / f"{name}_state")
        except Exception:  # logged with its traceback; the run goes on
            logger.exception("Writing the resume state for step %s failed; the previous state is kept.", step)
            remove_unfinished(output_dir / f"{name}_state.tmp", "resume state")
    return True


def save_state(host: FinetuneHost, state_dir: Path) -> None:
    """Everything train.resume_from needs, written aside and swapped in whole."""
    started = time.time()
    tmp = state_dir.with_name(state_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write_resume_weights(host.rotator, host.always_on, tmp / STATE_WEIGHTS)
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
            "steps_taken": host._steps_taken,
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
