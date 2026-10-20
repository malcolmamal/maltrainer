#!/usr/bin/env python3
"""
Krea 2 full finetuning for maltrainer: the DiT's own weights, on one 32 GB card.

Krea 2's 12.9B bf16 weights (25.8 GB) do not fit beside their training state, so the blocks
train a window at a time (maltrainer/finetune/rotation.py, after Fizgig):

- the frozen model stays on the GPU in fp8 (block Linears per output row), as for LoRAs
- one window (every block's attention, then every block's MLP gate, up, down) is trainable in
  bf16 from a CPU master copy; ``finetune.rotate_every`` steps later it is written back and the
  next window comes up. The text-fusion stack trains all run
- fused Automagic3 steps each weight inside backward (no full gradient copy) and rounds the
  update stochastically into bf16. Each window keeps its optimizer state in CPU RAM while frozen
- saves write a full bf16 ComfyUI file (~26 GB) plus a resume state

Data, caching, timesteps and loss are the Krea 2 LoRA trainer's (train_krea2_lora.py).

    python train_krea2_finetune.py --config templates/krea2_finetune_template.yaml
    python train_krea2_finetune.py --config my_run.yaml --resume-from output/.../krea2_ft_x_state
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch
from tqdm import tqdm

from maltrainer.finetune.rotation import WindowRotator
from maltrainer.models.krea2.finetune import DEFAULT_ALWAYS_ON, always_on_linears, parse_windows, save_krea2_model
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.optimizers import Automagic3
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.fsutil import replace_directory
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata
from maltrainer.utils.train_monitor import TrainMonitor
from train_krea2_lora import Krea2LoRATrainer, plan_training_steps

logger = logging.getLogger(__name__)

STATE_MODEL = "model.safetensors"
STATE_OPTIMIZER = "optimizer.pt"
STATE_TRAINER = "trainer_state.pt"


class Krea2FinetuneTrainer(Krea2LoRATrainer):
    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        if config.get("train", {}).get("dop"):
            raise ValueError("train.dop is the LoRA trainer's; the Krea 2 finetune has no prior preservation yet")
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "krea2_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("Krea 2 finetuning runs on one GPU")
        if int(config.get("train", {}).get("gradient_accumulation_steps", 1) or 1) != 1:
            raise ValueError("fused Automagic3 steps inside backward, so gradient_accumulation_steps must be 1")
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: dict[str, Any] | None = None
        self.global_step = 0
        self.epoch = 0
        self.position = 0
        self.epoch_rng: tuple[object, ...] | None = None
        self.always_optimizer: Automagic3 | None = None
        self.key_prefix = ""

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading Krea 2 DiT for finetuning ===", flush=True)
        model_config = self.config["model"]
        finetune_config = self.config.get("finetune", {}) or {}
        path = Path(model_config["transformer_path"])
        if self.resume_dir is not None:
            for name in (STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER):
                if not (self.resume_dir / name).exists():
                    raise FileNotFoundError(f"train.resume_from: {self.resume_dir / name} is missing")
            path = self.resume_dir / STATE_MODEL
            logger.info("Resuming from %s", self.resume_dir)
        master: dict[str, torch.Tensor] = {}
        self.transformer, info = load_krea2_transformer(path, self.device, self.dtype, master=master)
        self.key_prefix = str(info.get("key_prefix", ""))
        self.run_metadata["base_load"] = {k: v for k, v in info.items() if k != "key_prefix"}
        if info.get("source_fp8_layers") and self.resume_dir is None:
            logger.warning(
                "%s stores its blocks in fp8: the finetune starts from dequantized fp8 weights, so every block "
                "keeps fp8's rounding. Use the bf16 Raw model (krea2_raw_bf16.safetensors) for a real run.",
                path.name,
            )
        self.transformer.enable_gradient_checkpointing()
        self.transformer.train()

        self.rotator = WindowRotator(
            self.transformer,
            master,
            parse_windows(finetune_config.get("windows")),
            self.device,
            self.dtype,
            self._make_optimizer,
        )
        always = finetune_config.get("always_on", list(DEFAULT_ALWAYS_ON)) or []
        self.always_on = always_on_linears(self.transformer, always)
        # the inherited _training_step calls whatever lora_network holds
        self.lora_network = self.transformer
        windows = ", ".join(
            f"{n} {self.rotator.window_parameters(i) / 1e9:.2f}B" for i, n in enumerate(self.rotator.names)
        )
        always_count = sum(p.numel() for _, p in self.always_on)
        self.run_metadata.update(
            {
                "windows": self.rotator.names,
                "always_on": list(always),
                "rotate_every": self._rotate_every(),
                "rotating_parameters": sum(self.rotator.window_parameters(i) for i in range(len(self.rotator.windows))),
                "always_on_parameters": always_count,
            }
        )
        logger.info(
            "Windows: %s; always trained: %s (%.2fB); rotating every %d steps",
            windows,
            ", ".join(always) or "nothing",
            always_count / 1e9,
            self._rotate_every(),
        )

    def _rotate_every(self) -> int:
        return max(int((self.config.get("finetune", {}) or {}).get("rotate_every", 100) or 100), 1)

    def _automagic_kwargs(self) -> dict[str, Any]:
        train_config = self.config["train"]
        return {
            "lr": float(train_config.get("learning_rate", 1e-5)),
            "min_lr": float(train_config.get("automagic_min_lr", 1e-5)),
            "max_lr": float(train_config.get("automagic_max_lr", 1e-5)),
            "beta2": float(train_config.get("automagic_beta2", 0.999)),
            "eps": float(train_config.get("automagic_eps", 1e-30)),
            "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
            "weight_decay": float(train_config.get("weight_decay", 0.0)),
            "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
            "fused": True,
        }

    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3:
        return Automagic3([{"params": params, "name": name}], **self._automagic_kwargs())

    def setup_optimizer(self) -> None:
        name = str(self.config["train"].get("optimizer", "automagic3")).lower()
        if name not in {"automagic3", "automagic", "automagic_v3"}:
            raise ValueError(
                f"Krea 2 finetuning supports automagic3 only (got {name!r}): it rounds updates stochastically "
                "into the bf16 weights and steps inside backward"
            )
        self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **self._automagic_kwargs()}
        logger.info("Optimizer: %s", self.run_metadata["optimizer_resolved"])
        if self.always_on:
            self.always_optimizer = self._make_optimizer([p for _, p in self.always_on], "txtfusion")
        if self.resume_dir is not None:
            saved = torch.load(self.resume_dir / STATE_OPTIMIZER, map_location="cpu", weights_only=False)
            self.rotator.load_state_dict(saved["rotation"])
            if self.always_optimizer is not None and saved.get("always") is not None:
                self.always_optimizer.load_state_dict(saved["always"])
            self.resume_state = torch.load(self.resume_dir / STATE_TRAINER, map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------ training

    def _window_at(self, step: int) -> int:
        return (step // self._rotate_every()) % len(self.rotator.windows)

    def _dataset_fingerprint(self) -> str:
        root = self.config["dataset"]["path"]
        digest = hashlib.sha1(str(self.config["dataset"].get("repeats", 1)).encode("utf-8"), usedforsecurity=False)
        for key, caption in self.samples:
            digest.update(f"\x1e{Path(os.path.relpath(key, root)).as_posix()}\x1f{caption}".encode())
        return digest.hexdigest()

    def train(self) -> None:
        train_config = self.config["train"]
        save_every = int(train_config.get("save_every", 0) or 0)
        if not self.samples:
            raise ValueError("No usable images: every file in the dataset failed to load")
        epoch_size = len(self._epoch_order())
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(epoch_size, train_config)
        self.run_metadata.update(
            {
                "steps_per_epoch": steps_per_epoch,
                "planned_total_steps": total_steps,
                "dataset_samples": len(self.samples),
            }
        )

        resume = self.resume_state
        same_data = True
        if resume is not None:
            self.global_step, self.epoch, self.position = resume["global_step"], resume["epoch"], resume["position"]
            same_data = resume.get("dataset_fingerprint") == self._dataset_fingerprint()
            if same_data:
                logger.info(
                    "Resumed at step %d (epoch %d, sample %d of its order)",
                    self.global_step,
                    self.epoch + 1,
                    self.position,
                )
            else:
                if self.position > 0:
                    self.epoch += 1
                self.position = 0
                logger.warning(
                    "The dataset changed since the saved state (%s samples then, %d now); resuming at "
                    "step %d with a fresh epoch.",
                    resume.get("dataset_samples", "?"),
                    epoch_size,
                    self.global_step,
                )
            if self.global_step >= total_steps:
                logger.warning(
                    "The resumed run is already at step %d of %d; raise train.max_steps to continue.",
                    self.global_step,
                    total_steps,
                )
                return
            num_epochs = self.epoch + -(-(total_steps - self.global_step + self.position) // steps_per_epoch)

        logger.info(
            "Starting Krea 2 finetuning: %d epochs, %d total steps, %d windows of %d steps",
            num_epochs,
            total_steps,
            len(self.rotator.windows),
            self._rotate_every(),
        )
        progress_bar = tqdm(
            total=total_steps, initial=self.global_step, desc="Training Steps", disable=not self.is_main_process
        )
        keys = KeyCommands() if self.is_main_process else None
        monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100))
        stopped = False
        shift_logged = [False]
        loss = None
        epoch = self.epoch
        first = resume is not None
        while self.global_step < total_steps:
            num_epochs = max(num_epochs, epoch + 1)
            if resume is not None and first and same_data:
                random.setstate(resume["epoch_rng"])
            self.epoch, self.epoch_rng = epoch, random.getstate()
            order = self._epoch_order()
            start = self.position if first else 0
            if resume is not None and first:
                random.setstate(resume["python_rng"])
                torch.set_rng_state(resume["torch_rng"])
                if resume.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(resume["cuda_rng"])
            first = False
            for pos in range(start, len(order)):
                window = self._window_at(self.global_step)
                if window != self.rotator.current:
                    started = time.time()
                    self.rotator.activate(window)
                    monitor.exclude(time.time() - started)
                    logger.info(
                        "Step %d: training window %s (%.2fB, %.1fs to swap)",
                        self.global_step,
                        self.rotator.names[window],
                        self.rotator.window_parameters(window) / 1e9,
                        time.time() - started,
                    )
                loss, _, sigma = self._training_step(order[pos], shift_logged)
                cast(Any, loss).backward()  # fused: every trained weight steps here
                self.rotator.optimizer.step()  # applies the window's rate votes
                if self.always_optimizer is not None:
                    self.always_optimizer.step()
                monitor.record(loss.item(), float(sigma))
                self.global_step += 1
                self.position = pos + 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "window": self.rotator.names[window],
                        "epoch": f"{epoch + 1}/{num_epochs}",
                    }
                )
                self._log_step(self.global_step, total_steps, loss, epoch, num_epochs)
                monitor.step_done(self.global_step, total_steps, self.rotator.optimizer)
                if save_every > 0 and self.global_step % save_every == 0 and self.global_step < total_steps:
                    self.save_checkpoint(self.global_step)
                stopped = keys is not None and keys.handle(self.global_step, self.save_checkpoint)
                if keys is not None:
                    monitor.exclude(keys.take_paused_seconds())
                if self.global_step >= total_steps or stopped:
                    break
            if stopped:
                break
            if self.global_step < total_steps:
                self.position = 0
                epoch += 1

        progress_bar.close()
        if keys is not None:
            keys.close()
        if stopped:
            logger.info("Krea 2 finetuning stopped by key at step %d; resume from the _state folder.", self.global_step)
            return
        self.save_checkpoint("final")
        logger.info("Krea 2 finetuning completed!")

    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None:
        window = self.rotator.names[self.rotator.current] if self.rotator.current is not None else "-"
        self.progress_logger.info(
            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, Window: %s, LR: %.2e",
            step,
            total,
            int(step / total * 100),
            loss.item(),
            epoch + 1,
            num_epochs,
            window,
            self.rotator.optimizer.get_avg_learning_rate(),
        )

    # ------------------------------------------------------------------ saving

    DISK_MARGIN = 1 << 30

    def _name(self) -> str:
        name = self.config.get("output", {}).get("name") or "krea2_finetune"
        return name[: -len(".safetensors")] if name.endswith(".safetensors") else name

    def _model_bytes(self) -> int:
        return sum(t.numel() * 2 for t in self.transformer.state_dict().values())  # bf16, an upper bound

    def save_checkpoint(self, step: int | str) -> bool:
        """The full model and, with train.save_state, the resume state (its weights hard-linked to
        the checkpoint). A save that does not fit is skipped (the final one waits for space)."""
        output_config = self.config.get("output", {})
        output_dir = Path(output_config["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self._name()
        save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
        need = self._model_bytes() + self.DISK_MARGIN
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
        runtime = dict(
            self.run_metadata,
            saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            global_step=self.global_step,
            save_dtype="bfloat16",
        )
        metadata = build_metadata(
            self.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True))
        )
        started = time.time()
        try:
            save_krea2_model(self.transformer, self.rotator, save_path, self.key_prefix, metadata)
        except Exception as error:
            Path(str(save_path) + ".tmp").unlink(missing_ok=True)
            logger.error("Saving step %s failed (%s); the partial file is removed and training continues.", step, error)
            if step == "final":
                raise
            return False
        self.last_saved_path = str(save_path)
        logger.info("Saved finetuned model to %s (%.0fs)", save_path, time.time() - started)
        if bool(self.config["train"].get("save_state", True)):
            try:
                self._save_state(output_dir / f"{name}_state", save_path)
            except Exception:  # logged with its traceback; the run goes on
                shutil.rmtree(output_dir / f"{name}_state.tmp", ignore_errors=True)
                logger.exception("Writing the resume state for step %s failed; the previous state is kept.", step)
        return True

    def _save_state(self, state_dir: Path, weights: Path) -> None:
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
                "rotation": self.rotator.state_dict(),
                "always": self.always_optimizer.state_dict() if self.always_optimizer is not None else None,
            },
            tmp / STATE_OPTIMIZER,
        )
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "position": self.position,
                "epoch_rng": self.epoch_rng,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "dataset_fingerprint": self._dataset_fingerprint(),
                "dataset_samples": len(self._epoch_order()),
            },
            tmp / STATE_TRAINER,
        )
        replace_directory(tmp, state_dir)
        logger.info("Saved resume state (step %d) to %s (%.0fs)", self.global_step, state_dir, time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune the Krea 2 DiT in maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume-from", type=str, help="A <name>_state folder; overrides train.resume_from")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    Krea2FinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
