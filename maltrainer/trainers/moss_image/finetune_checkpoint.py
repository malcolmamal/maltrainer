"""The MOSS-Image finetune's model saves: the checkpoint in the source file's layout (the EMA average if on),
after waiting for disk space, with the resume state beside it."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import torch

from maltrainer.models.moss_image.finetune import save_full_model
from maltrainer.trainers.moss_image.finetune_state import FinetuneState
from maltrainer.utils.fsutil import remove_unfinished
from maltrainer.utils.lora_metadata import build_metadata

logger = logging.getLogger(__name__)


class FinetuneCheckpoint(FinetuneState):
    DISK_MARGIN = 1 << 30  # 1 GB left over after a save

    def _name(self) -> str:
        output_config = self.config.get("output", {})
        name = output_config.get("name") or "moss_image_finetune"
        return name[: -len(".safetensors")] if name.endswith(".safetensors") else name

    def _state_bytes(self, hard_linked: bool) -> int:
        """What a resume state writes: the raw weights (unless hard-linked to the checkpoint), the
        optimizer state, the fp32 EMA and the trainer counters."""
        total = 0 if hard_linked else self.layout_path.stat().st_size
        for state in self.optimizer.state.values():
            total += sum(t.numel() * t.element_size() for t in state.values() if torch.is_tensor(t))
        if self.ema is not None:
            total += sum(t.numel() * 4 for t in self.ema.shadow)
        return total + (16 << 20)

    @staticmethod
    def _free_bytes(folder: Path) -> int:
        return shutil.disk_usage(folder).free

    def save_checkpoint(self, step: int | str) -> bool:
        """The model (EMA average if on) and, with train.save_state, the resume state. Returns
        whether the model was saved. A full disk never ends the run: a save that does not fit is
        skipped (the final one waits for space), and a write that fails is cleaned up."""
        output_config = self.config.get("output", {})
        output_dir = Path(output_config["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self._name()
        save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
        save_dtype = getattr(torch, str(output_config.get("dtype", "bfloat16")))
        need = self.layout_path.stat().st_size + self.DISK_MARGIN
        while self._free_bytes(output_dir) < need:
            free = self._free_bytes(output_dir)
            if step != "final":
                logger.error(
                    "Not saving step %s: it needs %.1f GB and %s has %.1f GB free. Training continues; "
                    "free space for the next save.",
                    step,
                    need / 1e9,
                    output_dir,
                    free / 1e9,
                )
                return False
            logger.error(
                "The final model needs %.1f GB and %s has %.1f GB free: free space there; checking again "
                "every 30 s (the trained weights are in memory, nothing is lost while it waits).",
                need / 1e9,
                output_dir,
                free / 1e9,
            )
            time.sleep(30)
        runtime = dict(
            self.run_metadata,
            saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            global_step=self.global_step,
            save_dtype=str(save_dtype).replace("torch.", ""),
        )
        metadata = build_metadata(
            self.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True))
        )
        started = time.time()
        overrides = self.ema.averaged() if self.ema is not None else None
        try:
            save_full_model(
                self.transformer, self.layout_path, save_path, overrides=overrides, metadata=metadata, dtype=save_dtype
            )
        except Exception as error:  # a full disk mid-write (safetensors raises its own error type)
            Path(str(save_path) + ".tmp").unlink(missing_ok=True)
            logger.error("Saving step %s failed (%s); the partial file is removed and training continues.", step, error)
            if step == "final":
                raise
            return False
        self.last_saved_path = str(save_path)
        logger.info(
            "Saved finetuned model%s to %s (%.0fs)", " (EMA)" if overrides else "", save_path, time.time() - started
        )
        if bool(self.config["train"].get("save_state", True)):
            # Without EMA, in the training dtype, the checkpoint holds exactly the raw weights.
            same_weights = overrides is None and save_dtype == self.dtype
            state_need = self._state_bytes(hard_linked=same_weights) + self.DISK_MARGIN
            free = self._free_bytes(output_dir)
            if free < state_need:
                logger.error(
                    "Not writing the resume state for step %s: it needs %.1f GB and %.1f GB is free. The "
                    "previous state is kept (resume from there, or from this checkpoint as a new base).",
                    step,
                    state_need / 1e9,
                    free / 1e9,
                )
            else:
                try:
                    self._save_state(output_dir / f"{name}_state", weights=save_path if same_weights else None)
                except Exception:  # logged with its traceback; the run goes on
                    logger.exception(
                        "Writing the resume state for step %s failed; the partial copy is removed and "
                        "the previous state kept.",
                        step,
                    )
                    remove_unfinished(output_dir / f"{name}_state.tmp", "resume state")
        return True
