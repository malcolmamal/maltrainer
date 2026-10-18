#!/usr/bin/env python3
"""
MOSS-Image full finetuning for maltrainer.

Trains the DiT's own weights (all 5.08B, or the subset finetune.trainable_prefixes names)
instead of a LoRA. Data, caching, timesteps and loss are the LoRA trainer's
(train_moss_image_lora.py); what differs is the model side:

- the bf16 DiT trains in place, its updates stochastically rounded into bf16 by Automagic3
- Automagic3 runs fused by default: each parameter steps inside backward the moment its gradient
  is ready, so no full set of gradients is ever held. Measured on an RTX 5090: 16.8 GB peak at
  1024 px, 19.0 GB at 1536 px. Fused means no gradient accumulation and no global gradient
  clipping; Automagic3's per-tensor RMS clip (automagic_clip_threshold) bounds each update.
- the optional EMA keeps its fp32 average (20 GB) in CPU RAM
- checkpoints are full ComfyUI single files in the source file's layout (~10 GB each)
- each save also writes <name>_state/ (raw weights, optimizer, EMA, step and RNG), which
  train.resume_from continues from exactly. Only the latest state is kept.
- by default only the Linear weights of the 42 joint blocks train (finetune.trainable_prefixes,
  finetune.linears_only); the text side stays frozen
- optional prior preservation (prior.weight, DOP) and regularisation images (dataset.reg_path),
  from maltrainer/finetune/

    python train_moss_image_finetune.py --config templates/moss_image_finetune_template.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from maltrainer.data.minimax_dataset import scan_media_sources
from maltrainer.finetune.prior import PriorPreservation, class_caption, resolve_classes, weights_from_file
from maltrainer.finetune.regularisation import set_step_scale
from maltrainer.finetune.repeats import folder_repeats, repeat_order, samples_per_epoch, unknown_keys
from maltrainer.finetune.selection import select_trainable
from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.moss_image.finetune import DEFAULT_TRAINABLE_PREFIXES, drift_targets, param_groups, save_full_model
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.optimizers import Automagic3
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.ema import OffloadedParameterEMA
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.lora_metadata import build_metadata
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.step_planner import DEFAULT_EPOCHS, Group, folder_counts
from maltrainer.utils.train_monitor import TrainMonitor, WeightDrift
from train_moss_image_lora import MOSSImageLoRATrainer, plan_training_steps

logger = logging.getLogger(__name__)

STATE_MODEL = "model.safetensors"
STATE_OPTIMIZER = "optimizer.pt"
STATE_EMA = "ema.safetensors"
STATE_TRAINER = "trainer_state.pt"
STATE_PRIOR = "prior.safetensors"


class MOSSImageFinetuneTrainer(MOSSImageLoRATrainer):
    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "moss_image_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("MOSS-Image finetuning runs on one GPU")
        self.ema: Optional[OffloadedParameterEMA] = None
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: Optional[dict] = None
        # Where the loop stands, for the resume state: the epoch, the next index into its
        # shuffled order, and the Python RNG state that shuffled it.
        self.global_step = 0
        self.epoch = 0
        self.position = 0
        self.epoch_rng = None
        # dataset.reg_path: regularisation images, trained at train.reg_lr_multiplier x the step
        self.reg_keys: Set[str] = set()
        self.reg_multiplier = float(config.get("train", {}).get("reg_lr_multiplier", 0.2))
        # prior.weight > 0: prior preservation against the original model (maltrainer/finetune/prior.py)
        prior_config = config.get("prior", {}) or {}
        self.prior: Optional[PriorPreservation] = None
        self.prior_classes = {}
        self.prior_path: Optional[Path] = None
        if float(prior_config.get("weight", 0.0) or 0.0) > 0:
            self.prior_classes = resolve_classes(prior_config, config.get("dataset", {}).get("trigger"))
            self.prior = PriorPreservation(
                weight=float(prior_config["weight"]),
                samples_per_image=int(prior_config.get("samples_per_image", 8)),
                every=int(prior_config.get("every", 1)),
                seed=int(config.get("seed", 42)),
            )

    # ------------------------------------------------------------------ data

    def _sources(self):
        sources = super()._sources()
        dataset_config = self.config["dataset"]
        reg_path = dataset_config.get("reg_path")
        if not reg_path:
            return sources
        reg = scan_media_sources(reg_path, kinds=["image"], default_caption=dataset_config.get("reg_default_caption", ""))
        if not reg:
            raise ValueError(f"dataset.reg_path {reg_path} has no images")
        self.reg_keys = {str(source.path) for source in reg}
        logger.info("Regularisation: %d images from %s at x%g the step", len(reg), reg_path, self.reg_multiplier)
        return sources + reg

    @classmethod
    def step_groups(cls, config: dict) -> List[Group]:
        groups = super().step_groups(config)
        dataset_config = config["dataset"]
        groups[0].folder_repeats = dict(dataset_config.get("folder_repeats") or {})
        reg_path = dataset_config.get("reg_path")
        if reg_path:
            groups.append(Group("Regularisation", reg_path, max(int(dataset_config.get("reg_repeats", 1)), 0), folder_counts(reg_path)))
        return groups

    @classmethod
    def step_notes(cls, config: dict) -> List[str]:
        """Folders prior preservation would miss. People are named by their folder through
        prefix.txt's [folder] ("photo of feliciaday, ..."), so a folder whose captions name its
        own folder is a person, and needs a prior.classes entry. Folders that name nobody (e.g.
        body/eyes: "extreme close-up photo of human eyes") are concepts and get no prior term.
        Folders with no caption at all would train on the empty prompt: they are named too."""
        dataset_config = config["dataset"]
        root = dataset_config.get("path", "")
        stray = unknown_keys(dataset_config.get("folder_repeats"), [name for name, _ in folder_counts(root)])
        stray_note = ([f"dataset.folder_repeats names {', '.join(stray)}, which matches no folder (no effect)"]
                      if stray else [])
        sources = scan_media_sources(root, kinds=["image"], trigger_token=dataset_config.get("trigger"),
                                     default_caption=dataset_config.get("default_caption", ""))
        by_folder: dict = {}
        for source in sources:
            by_folder.setdefault(source.path.parent, []).append(source.caption)
        notes = list(stray_note)
        uncaptioned = sorted(Path(os.path.relpath(folder, root)).as_posix()
                             for folder, captions in by_folder.items() if any(not c.strip() for c in captions))
        if uncaptioned:
            notes.append(f"images with no caption at all in {', '.join(uncaptioned)}: they would train on the empty "
                         f"prompt (CFG's negative); add .txt captions or a prefix.txt")
        prior_config = config.get("prior", {}) or {}
        if float(prior_config.get("weight", 0.0) or 0.0) <= 0:
            return notes
        try:
            classes = {k.lower() for k in resolve_classes(prior_config, dataset_config.get("trigger"))}
        except ValueError as error:
            return notes + [str(error)]
        def names_its_folder(folder: Path, captions: List[str]) -> bool:
            # the nearest prefix.txt decides, as it does for the captions: "[folder]" in it names
            # the person; fixed text ("extreme close-up photo of human eyes") is a concept
            stop = Path(root).resolve().parent
            current = folder.resolve()
            while True:
                prefix = current / "prefix.txt"
                if prefix.exists():
                    return "[folder]" in prefix.read_text(encoding="utf-8")
                if current == stop or current.parent == current:
                    break
                current = current.parent
            # no prefix.txt: a person only if the captions spell the folder's name out
            return any(re.search(rf"(?<!\w){re.escape(folder.name)}(?!\w)", c, re.IGNORECASE) for c in captions)

        people = {folder.name.lower(): Path(os.path.relpath(folder, root)).as_posix()
                  for folder, captions in by_folder.items() if names_its_folder(folder, captions)}
        missing = sorted(rel for name, rel in people.items() if name not in classes)
        if missing:
            notes.append(f"prior.classes has no entry for {', '.join(missing)}: add each as "
                         f"<folder>: woman / man, or those people train without prior preservation")
        unused = sorted(classes - set(people))
        if people and unused:
            notes.append(f"prior.classes names {', '.join(unused)}, which no folder here names (harmless)")
        return notes

    def _dry_run_samples(self) -> int:
        """One epoch, counted from the folders as --suggest-steps does (per-folder repeats included)."""
        return sum(group.samples for group in self.step_groups(self.config))

    def _rel(self, key: str) -> str:
        """A sample's name relative to its dataset folder, so a moved dataset still matches."""
        root = self.config["dataset"].get("reg_path") if key in self.reg_keys else self.config["dataset"]["path"]
        return Path(os.path.relpath(key, root)).as_posix()

    def _prior_captions(self) -> dict:
        """sample key -> its class caption, for every subject image whose caption names a trigger."""
        if self.prior is None:
            return {}
        out = {}
        for key, caption in self.samples:
            if key in self.reg_keys:
                continue
            replaced = class_caption(caption, self.prior_classes)
            if replaced is not None:
                out[key] = replaced
        return out

    def _extra_prompts(self) -> List[str]:
        return super()._extra_prompts() + sorted(set(self._prior_captions().values()))

    def _repeat_groups(self) -> List[tuple]:
        """(sample indices, repeats) per subject folder (dataset.folder_repeats, else
        dataset.repeats), and the regularisation images with dataset.reg_repeats."""
        dataset_config = self.config["dataset"]
        default = float(dataset_config.get("repeats", 1) or 1)
        table = dataset_config.get("folder_repeats") or {}
        folders: Dict[str, List[int]] = {}
        reg = []
        for i, (key, _) in enumerate(self.samples):
            if key in self.reg_keys:
                reg.append(i)
            else:
                folders.setdefault(Path(os.path.relpath(Path(key).parent, dataset_config["path"])).as_posix(), []).append(i)
        groups = [(indices, folder_repeats(folder, table, default)) for folder, indices in sorted(folders.items())]
        return groups + [(reg, float(max(dataset_config.get("reg_repeats", 1), 0)))]

    def _epoch_order(self) -> List[int]:
        return repeat_order(self._repeat_groups())

    def _epoch_size(self) -> int:
        return sum(samples_per_epoch(len(indices), repeats) for indices, repeats in self._repeat_groups())

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self):
        print("=== Step 2/2: Loading MOSS-Image DiT for finetuning ===", flush=True)
        train_config = self.config.get("train", {})
        finetune_config = self.config.get("finetune", {}) or {}
        # Saves copy the key layout of model.transformer_path; on resume the weights come from
        # the state folder, which the next save replaces, so it cannot serve as the layout.
        self.layout_path = Path(self.config["model"]["transformer_path"])
        if is_int8_convrot(self.layout_path):
            raise ValueError(
                f"{self.layout_path.name} is int8; finetuning needs the plain bf16 DiT "
                "(moss_image_base_bf16.safetensors, or a model this trainer saved)"
            )
        weights_path = self.layout_path
        if self.resume_dir is not None:
            for name in (STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER):
                if not (self.resume_dir / name).exists():
                    raise FileNotFoundError(f"train.resume_from: {self.resume_dir / name} is missing")
            weights_path = self.resume_dir / STATE_MODEL
            logger.info("Resuming from %s", self.resume_dir)

        self.transformer, load_info = load_moss_image_transformer(weights_path, self.device, self.dtype)
        self.run_metadata["base_load"] = load_info
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()

        # No key = the MOSS-Image default (the blocks); an explicit [] = everything.
        prefixes = finetune_config.get("trainable_prefixes")
        prefixes = list(DEFAULT_TRAINABLE_PREFIXES) if prefixes is None else list(prefixes)
        linears_only = bool(finetune_config.get("linears_only", True))
        self.named_trainable = select_trainable(self.transformer, prefixes, linears_only=linears_only)
        self.transformer.train()
        # The inherited _training_step runs whatever lora_network holds.
        self.lora_network = self.transformer
        trainable = sum(p.numel() for _, p in self.named_trainable)
        total = sum(p.numel() for p in self.transformer.parameters())
        self.run_metadata.update({
            "trainable_parameters": trainable,
            "trainable_prefixes": list(prefixes) or "all",
            "linears_only": linears_only,
        })
        logger.info(
            "Finetuning %.3fB of %.3fB parameters (%s%s)", trainable / 1e9, total / 1e9,
            ", ".join(prefixes) or "all", ", Linear weights only" if linears_only else "",
        )
        self._maybe_compile()

    def setup_optimizer(self):
        train_config = self.config["train"]
        name = str(train_config.get("optimizer", "automagic3")).lower()
        if name not in {"automagic3", "automagic", "automagic_v3"}:
            raise ValueError(
                f"MOSS-Image finetuning supports automagic3 only (got {name!r}): it rounds updates "
                "stochastically into the bf16 weights and keeps ~1 byte of state per parameter"
            )
        fused = bool(train_config.get("automagic_fused", True))
        accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
        if fused and accum > 1:
            raise ValueError("train.automagic_fused steps inside backward, so gradient_accumulation_steps must be 1")
        groups = param_groups(self.named_trainable, bool(train_config.get("automagic_param_groups", True)))
        kwargs = {
            "lr": float(train_config.get("learning_rate", 1e-6)),
            "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
            "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
            "beta2": float(train_config.get("automagic_beta2", 0.999)),
            "eps": float(train_config.get("automagic_eps", 1e-30)),
            "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
            "weight_decay": float(train_config.get("weight_decay", 0.0)),
            "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
            "fused": fused,
        }
        self.optimizer = Automagic3(groups, **kwargs)
        self.run_metadata["optimizer_resolved"] = {"name": "automagic3", "groups": [g["name"] for g in groups], **kwargs}
        logger.info("Optimizer: %s", self.run_metadata["optimizer_resolved"])

        if self.resume_dir is not None:
            state = torch.load(self.resume_dir / STATE_TRAINER, map_location="cpu", weights_only=False)
            saved_groups = state.get("group_names")
            if saved_groups != [g["name"] for g in groups]:
                raise ValueError(
                    f"The resumed optimizer had groups {saved_groups}, this config builds "
                    f"{[g['name'] for g in groups]}; keep finetune.trainable_prefixes and "
                    "train.automagic_param_groups as they were"
                )
            self.optimizer.load_state_dict(torch.load(self.resume_dir / STATE_OPTIMIZER, map_location=self.device, weights_only=False))
            self.resume_state = state

    # ------------------------------------------------------------------ training

    def train(self):
        train_config = self.config["train"]
        save_every = int(train_config.get("save_every", 0) or 0)
        accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)

        ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
        if ema_decay > 0:
            self.ema = OffloadedParameterEMA(
                self.named_trainable, decay=ema_decay, warmup=bool(train_config.get("ema_warmup", True)),
                every=int(train_config.get("ema_every", 1) or 1),
            )
            self.run_metadata.update({"ema_decay": ema_decay, "ema_every": self.ema.every})
            logger.info("EMA in CPU RAM: decay %.4f every %d step(s) (checkpoints save the average)", ema_decay, self.ema.every)

        if not self.samples:
            raise ValueError("No usable images: every file in the dataset failed to load")
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(self._epoch_size(), train_config)
        self.run_metadata.update({"steps_per_epoch": steps_per_epoch, "planned_total_steps": total_steps, "dataset_samples": len(self.samples)})

        resume = self.resume_state
        same_data = True
        if resume is not None:
            self.global_step, self.epoch, self.position = resume["global_step"], resume["epoch"], resume["position"]
            if self.ema is not None:
                if not (self.resume_dir / STATE_EMA).exists():
                    logger.warning("The resumed run had no EMA; the average starts from the current weights.")
                else:
                    self.ema.load_state(load_file(str(self.resume_dir / STATE_EMA)), resume.get("ema_counters", {}))
            # Older states carry no fingerprint; treat them as the same data.
            same_data = resume.get("dataset_fingerprint", self._dataset_fingerprint()) == self._dataset_fingerprint()
            if same_data:
                logger.info("Resumed at step %d (epoch %d, sample %d of its order)", self.global_step, self.epoch + 1, self.position)
            else:
                # The old epoch's shuffled order means nothing for a different set: carry on
                # with a fresh epoch. Weights, optimizer, EMA and step count still continue.
                if self.position > 0:
                    self.epoch += 1
                self.position = 0
                logger.warning(
                    "The dataset changed since the saved state (%s samples then, %d now); resuming at step %d "
                    "with a fresh epoch.", resume.get("dataset_samples", "?"), self._epoch_size(), self.global_step,
                )
            if self.global_step >= total_steps:
                logger.warning("The resumed run is already at step %d of %d; raise train.max_steps to continue.", self.global_step, total_steps)
                return
        self._setup_prior()
        if self.reg_keys:
            self.run_metadata.update({"reg_images": len(self.reg_keys), "reg_lr_multiplier": self.reg_multiplier})

        logger.info(
            "Starting MOSS-Image finetuning: %d epochs, %d total steps; timesteps %s, shift %.2f",
            num_epochs, total_steps, train_config.get("timestep_distribution", "logit_normal"),
            float(train_config.get("timestep_shift", 1.0)),
        )
        progress_bar = tqdm(total=total_steps, initial=self.global_step, desc="Training Steps", disable=not self.is_main_process)
        keys = KeyCommands(menu=self._preview_menu)
        stopped = False
        micro = 0
        loss = None
        scales: List[float] = []
        prior_losses: List[float] = []
        # train.monitor_every: a health summary line (loss by noise band, prior, the optimizer's
        # direction signal, weight drift, NaN counts, speed, VRAM); maltrainer/utils/train_monitor.py
        drift = None
        if bool(train_config.get("monitor_drift", True)):
            tracked = drift_targets(self.named_trainable)
            if tracked:
                drift = WeightDrift(tracked)
                logger.info("Monitor: weight drift of %d tensors (%.0f MB of copies)", len(tracked), drift.size_mb)
        monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100), drift=drift)
        if resume is None:
            monitor.exclude(self._maybe_preview(0, total_steps))  # the base model: what later previews compare with
        epoch = self.epoch
        first = resume is not None
        # Until the step budget is spent: a resumed run may need more epochs than num_epochs
        # suggests, e.g. after a dataset change.
        while self.global_step < total_steps:
            num_epochs = max(num_epochs, epoch + 1)
            if first and same_data:
                random.setstate(resume["epoch_rng"])
            self.epoch, self.epoch_rng = epoch, random.getstate()
            order = self._epoch_order()
            start = self.position if first else 0
            if first:
                random.setstate(resume["python_rng"])
                torch.set_rng_state(resume["torch_rng"])
                if resume.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(resume["cuda_rng"])
            first = False
            for pos in range(start, len(order)):
                index = order[pos]
                key = self.samples[index][0]
                loss = self._training_step(index)
                total_loss = loss
                prior_value = None
                if self.prior is not None and self.prior.due(self.global_step):
                    x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
                    prior_loss = self.prior.loss(self._rel(key), x0, self._predict)
                    if prior_loss is not None:
                        # One backward for both terms: fused Automagic3 steps inside it.
                        total_loss = loss + self.prior.weight * prior_loss
                        prior_value = prior_loss.item()
                        prior_losses.append(prior_value)
                scales.append(self.reg_multiplier if key in self.reg_keys else 1.0)
                if self.reg_keys and self.optimizer.fused:
                    set_step_scale(self.optimizer, scales[-1])  # the update happens inside this backward
                (total_loss / accum if accum > 1 else total_loss).backward()
                # Regularisation images count in the loss bands too; they are a fifth of the step.
                monitor.record(loss.item(), float(self.last_sigma), prior_value)
                micro += 1
                if micro % accum:
                    continue
                if self.reg_keys and not self.optimizer.fused:
                    set_step_scale(self.optimizer, sum(scales) / len(scales))
                scales.clear()
                self.step_note = "Prior: %.6f, " % (sum(prior_losses) / len(prior_losses)) if prior_losses else ""
                prior_losses.clear()
                # Fused: every parameter already stepped in backward; this applies the
                # per-group rate votes. Unfused: the whole update happens here.
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update()

                self.global_step += 1
                self.position = pos + 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch + 1}/{num_epochs}"})
                self._log_step(self.global_step, total_steps, loss, epoch, num_epochs)
                monitor.step_done(self.global_step, total_steps, self.optimizer)
                monitor.exclude(self._maybe_preview(self.global_step, total_steps))
                if save_every > 0 and self.global_step % save_every == 0 and self.global_step < total_steps:
                    self.save_checkpoint(self.global_step)
                stopped = keys is not None and keys.handle(self.global_step, self.save_checkpoint)
                if keys is not None:
                    monitor.exclude(keys.take_paused_seconds())
                self._watch_config()
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
            logger.info("MOSS-Image finetuning stopped by key at step %d; resume from the _state folder.", self.global_step)
            return
        self.save_checkpoint("final")
        logger.info("MOSS-Image finetuning completed!")

    def _preview_variants(self):
        """With EMA on, two rows: the raw weights (what is training now) and the average (what
        checkpoints save). The noise of batch-1 steps shows in the first and averages out in the
        second."""
        if self.ema is None:
            return [("", contextlib.nullcontext)]
        return [("raw", contextlib.nullcontext), ("ema", self.ema.applied)]

    def _predict(self, noisy: torch.Tensor, sigma: torch.Tensor, caption: str) -> torch.Tensor:
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        return self.transformer(noisy.to(self.dtype), sigma, context)

    def _setup_prior(self) -> None:
        """Record (or, on resume, reload) the original model's predictions for the prior term."""
        if self.prior is None:
            return
        captions = self._prior_captions()
        if not captions:
            raise ValueError(f"prior.weight > 0, but no caption names a trigger of prior.classes {sorted(self.prior_classes)}")
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
                logger.info("Prior: %d image(s) added since it was recorded; recording theirs from the original "
                            "weights in %s (~%.0f min)", len(missing), self.layout_path.name,
                            len(missing) * self.prior.samples_per_image / 6 / 60)
                was_training = self.transformer.training
                self.transformer.eval()
                with weights_from_file(self.named_trainable, self.layout_path):
                    self.prior.build(
                        ((self._rel(key), self.latent_cache[key], caption) for key, caption in missing.items()),
                        self._sample_sigma, self._predict, self.device,
                    )
                self.transformer.train(was_training)
                torch.cuda.empty_cache()
        else:
            self.prior.build(
                ((self._rel(key), self.latent_cache[key], caption) for key, caption in captions.items()),
                self._sample_sigma, self._predict, self.device,
            )
        self.prior_path = Path(self.config["output"]["path"]) / f"{self._name()}_prior.safetensors"
        self.prior_path.parent.mkdir(parents=True, exist_ok=True)
        self.prior.save(self.prior_path)
        self.run_metadata.update({
            "prior_weight": self.prior.weight,
            "prior_samples_per_image": self.prior.samples_per_image,
            "prior_every": self.prior.every,
            "prior_classes": self.prior_classes,
            "prior_images": len(self.prior),
        })
        logger.info("Prior preservation: weight %g on %d images, every %d step(s), classes %s",
                    self.prior.weight, len(self.prior), self.prior.every, self.prior_classes)

    # ------------------------------------------------------------------ saving

    def _dataset_fingerprint(self) -> str:
        """The samples (paths relative to the dataset folder, captions) and repeats, so a moved
        dataset still matches and any added, removed or recaptioned image does not."""
        root = self.config["dataset"]["path"]
        digest = hashlib.sha1(str(self.config["dataset"].get("repeats", 1)).encode("utf-8"))
        if self.config["dataset"].get("folder_repeats"):  # only then, so earlier states keep their fingerprint
            digest.update(repr(sorted(self.config["dataset"]["folder_repeats"].items())).encode("utf-8"))
        for key, caption in self.samples:
            if key in self.reg_keys:  # marked only then, so states without regularisation keep their fingerprint
                digest.update(f"\x1ereg:{self._rel(key)}\x1f{caption}".encode("utf-8"))
            else:
                digest.update(f"\x1e{Path(os.path.relpath(key, root)).as_posix()}\x1f{caption}".encode("utf-8"))
        return digest.hexdigest()

    def _name(self) -> str:
        output_config = self.config.get("output", {})
        name = output_config.get("name") or "moss_image_finetune"
        return name[: -len(".safetensors")] if name.endswith(".safetensors") else name

    # ------------------------------------------------------------------ disk space

    DISK_MARGIN = 1 << 30  # 1 GB left over after a save

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

    def save_checkpoint(self, step) -> bool:
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
                logger.error("Not saving step %s: it needs %.1f GB and %s has %.1f GB free. Training continues; "
                             "free space for the next save.", step, need / 1e9, output_dir, free / 1e9)
                return False
            logger.error("The final model needs %.1f GB and %s has %.1f GB free: free space there; checking again "
                         "every 30 s (the trained weights are in memory, nothing is lost while it waits).",
                         need / 1e9, output_dir, free / 1e9)
            time.sleep(30)
        runtime = dict(self.run_metadata, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), global_step=self.global_step,
                       save_dtype=str(save_dtype).replace("torch.", ""))
        metadata = build_metadata(self.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True)))
        started = time.time()
        overrides = self.ema.averaged() if self.ema is not None else None
        try:
            save_full_model(self.transformer, self.layout_path, save_path, overrides=overrides, metadata=metadata, dtype=save_dtype)
        except Exception as error:  # a full disk mid-write (safetensors raises its own error type)
            Path(str(save_path) + ".tmp").unlink(missing_ok=True)
            logger.error("Saving step %s failed (%s); the partial file is removed and training continues.", step, error)
            if step == "final":
                raise
            return False
        self.last_saved_path = str(save_path)
        logger.info("Saved finetuned model%s to %s (%.0fs)", " (EMA)" if overrides else "", save_path, time.time() - started)
        if bool(self.config["train"].get("save_state", True)):
            # Without EMA, in the training dtype, the checkpoint holds exactly the raw weights.
            same_weights = overrides is None and save_dtype == self.dtype
            state_need = self._state_bytes(hard_linked=same_weights) + self.DISK_MARGIN
            free = self._free_bytes(output_dir)
            if free < state_need:
                logger.error("Not writing the resume state for step %s: it needs %.1f GB and %.1f GB is free. The "
                             "previous state is kept (resume from there, or from this checkpoint as a new base).",
                             step, state_need / 1e9, free / 1e9)
            else:
                try:
                    self._save_state(output_dir / f"{name}_state", weights=save_path if same_weights else None)
                except Exception as error:
                    shutil.rmtree(output_dir / f"{name}_state.tmp", ignore_errors=True)
                    logger.error("Writing the resume state for step %s failed (%s); the partial copy is removed and "
                                 "the previous state kept.", step, error)
        return True

    def _save_state(self, state_dir: Path, weights: Optional[Path] = None) -> None:
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
            save_full_model(self.transformer, self.layout_path, tmp / STATE_MODEL, metadata={"maltrainer_state_step": str(self.global_step)})
        torch.save(self.optimizer.state_dict(), tmp / STATE_OPTIMIZER)
        if self.prior is not None and self.prior_path is not None:
            try:
                os.link(self.prior_path, tmp / STATE_PRIOR)
            except OSError:
                shutil.copyfile(self.prior_path, tmp / STATE_PRIOR)
        ema_counters = None
        if self.ema is not None:
            tensors, ema_counters = self.ema.state()
            save_file(tensors, str(tmp / STATE_EMA))
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "position": self.position,
            "epoch_rng": self.epoch_rng,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "ema_counters": ema_counters,
            "group_names": [g["name"] for g in self.optimizer.param_groups],
            "dataset_fingerprint": self._dataset_fingerprint(),
            "dataset_samples": self._epoch_size(),
        }, tmp / STATE_TRAINER)
        old = state_dir.with_name(state_dir.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if state_dir.exists():
            state_dir.rename(old)
        tmp.rename(state_dir)
        shutil.rmtree(old, ignore_errors=True)
        logger.info("Saved resume state (step %d%s) to %s (%.0fs)", self.global_step,
                    ", weights hard-linked to the checkpoint" if linked else "", state_dir, time.time() - started)


def main():
    parser = argparse.ArgumentParser(description="Finetune the MOSS-Image DiT in maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume-from", type=str, help="A <name>_state folder; overrides train.resume_from")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    parser.add_argument("--suggest-steps", type=float, nargs="?", const=DEFAULT_EPOCHS, metavar="EPOCHS",
                        help=f"Count the subject and regularisation images and suggest train.max_steps for "
                             f"EPOCHS epochs of each (default {DEFAULT_EPOCHS}), then exit without training")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.suggest_steps is not None:
        print(MOSSImageFinetuneTrainer.suggest_steps(config, args.suggest_steps))
        return
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    trainer = MOSSImageFinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.config_path = Path(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
