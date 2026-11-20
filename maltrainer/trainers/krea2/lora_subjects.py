"""Krea 2 LoRA of up to four subjects in one file, each kept to its own trigger (Fizgig LoRA Isolation).

``isolation.subjects`` lists the folders and triggers. On top of the masked LoRA trainer this adds:

- the trigger joined to what it is in every caption (``zwxem, a woman`` -> ``zwxemwoman``), so the subject never
  shares the plain class token every other person uses;
- DOP for every subject at once (each joined word swapped for its class phrase in the trigger-free caption);
- ``balance``: smaller folders repeat so each subject trains about as often as the largest;
- ``side_by_side``: a few pictures of each person next to a stranger, trained at a higher rate on their own steps.

Only the LoRA entry point uses this class; the finetune trainer never does.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch

from maltrainer.data.composites import build_composites, load_strangers
from maltrainer.data.minimax_dataset import MediaSource
from maltrainer.data.subject_text import dop_caption
from maltrainer.data.subjects import Subject, balanced_repeats, load_subjects
from maltrainer.optimizers import Automagic3
from maltrainer.trainers.krea2.lora_masked import MaskedKrea2LoRATrainer
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config

logger = logging.getLogger(__name__)


def with_subject_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """The config an isolation run starts from: the first subject as the dataset (the single-subject code reads
    ``dataset.path`` and ``dataset.trigger``), and DOP on at Fizgig's setting (every second step, weight 2) unless
    the config says otherwise. ``train.dop_class`` is only a placeholder for that code's check: every subject's own
    class is used."""
    isolation = config.get("isolation") or {}
    entries = isolation.get("subjects") or []
    if not entries:
        return config
    config = copy.deepcopy(config)
    first = entries[0]
    config.setdefault("dataset", {}).update(
        {"path": str(first.get("path", "")), "trigger": str(first.get("trigger", ""))}
    )
    train = config.setdefault("train", {})
    train.setdefault("dop", True)
    train.setdefault("dop_every", 2)
    train.setdefault("dop_multiplier", 2.0)
    train.setdefault("dop_class", "subject")
    return config


class Krea2SubjectsLoRATrainer(MaskedKrea2LoRATrainer):
    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        super().__init__(with_subject_defaults(config), dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.isolation: dict[str, Any] = self.config.get("isolation") or {}
        self.subjects: list[Subject] = []
        self.owner: dict[str, int] = {}  # sample key -> its subject's index
        self.composites: set[str] = set()
        self.lr_boost = 1.0
        self._step_boost = 1.0
        self._boostable = False

    # ------------------------------------------------------------------ data

    def _sources(self) -> list[MediaSource]:
        if not self.isolation.get("subjects"):
            return super()._sources()
        self.subjects = load_subjects(self.isolation, str(self.config["dataset"].get("default_caption", "")))
        sources: list[MediaSource] = []
        for index, subject in enumerate(self.subjects):
            sources += subject.sources
            self.owner.update({str(s.path): index for s in subject.sources})
        return sources + self._side_by_side()

    def _side_by_side(self) -> list[MediaSource]:
        options = self.isolation.get("side_by_side") or {}
        if not options.get("enabled", False):
            return []
        strangers = load_strangers(options.get("strangers_path", ""))
        folder = Path(self.config["output"]["path"]) / "side_by_side"
        self.lr_boost = float(options.get("lr_boost", 2.0))
        made: list[MediaSource] = []
        for index, subject in enumerate(self.subjects):
            if subject.words.kind == "person":
                made += build_composites(
                    subject,
                    index,
                    strangers,
                    folder,
                    int(options.get("per_subject", 16)),
                    int(self.config.get("seed", 42)),
                    int(options.get("panel", 512)),
                )
        self.composites = {str(s.path) for s in made}
        logger.info("Isolation: %d side-by-side pictures (trained at x%g on their own steps)", len(made), self.lr_boost)
        return made

    def _dry_run_samples(self) -> int:
        if not self.isolation.get("subjects"):
            return super()._dry_run_samples()
        """One epoch as the run will have it: balanced subjects plus the side-by-sides, times ``dataset.repeats``."""
        subjects = load_subjects(self.isolation, "")
        counts = [len(s.sources) for s in subjects]
        repeats = balanced_repeats(counts) if self.isolation.get("balance", True) else [1] * len(counts)
        options = self.isolation.get("side_by_side") or {}
        people = sum(s.words.kind == "person" for s in subjects) if options.get("enabled", False) else 0
        total = sum(c * r for c, r in zip(counts, repeats, strict=True)) + people * int(options.get("per_subject", 16))
        return total * max(int(self.config["dataset"].get("repeats", 1) or 1), 1)

    def precache_dataset(self) -> None:
        super().precache_dataset()
        if self.subjects and self.isolation.get("balance", True):
            self._balance()

    def _balance(self) -> None:
        counts = [sum(1 for key, _ in self.samples if self.owner.get(key) == i) for i in range(len(self.subjects))]
        repeats = balanced_repeats(counts)
        extra = []
        for pair in self.samples:
            if pair[0] in self.owner:  # the side-by-sides belong to no subject's count
                extra += [pair] * (repeats[self.owner[pair[0]]] - 1)
        self.samples += extra
        self.run_metadata["isolation_balance"] = repeats
        logger.info("Isolation: subjects repeat %s times per epoch (%s images)", repeats, counts)

    # ------------------------------------------------------------------ DOP

    def _plan_dop_captions(self) -> None:
        if not self.subjects:
            super()._plan_dop_captions()
            return
        pairs = [(s.word, s.words.phrase) for s in self.subjects]
        found = {key: out for key, caption in self.samples if (out := dop_caption(caption, pairs)) is not None}
        if not found:
            raise ValueError("train.dop: no caption contains any subject's trigger")
        missing = len(self.samples) - len(found)
        if missing:
            logger.warning("DOP: %d caption(s) name no subject and get no DOP pass.", missing)
        self.dop_captions = found
        train = self.config["train"]
        self.run_metadata["dop"] = {
            "subjects": [{"trigger": s.words.trigger, "word": s.word, "class": s.words.phrase} for s in self.subjects],
            "multiplier": float(train.get("dop_multiplier", 1.0)),
            "every": int(train.get("dop_every", 1) or 1),
            "captions": len(found),
        }
        logger.info(
            "DOP: %d of %d captions name a subject (e.g. %s)", len(found), len(self.samples), next(iter(found.values()))
        )

    # ------------------------------------------------------------------ the higher rate on the side-by-sides

    def setup_optimizer(self) -> None:
        super().setup_optimizer()
        self._boostable = not isinstance(self.optimizer, Automagic3)
        if self.composites and not self._boostable:
            logger.warning(
                "isolation.side_by_side.lr_boost is ignored with Automagic3 (it adapts every weight's own rate); "
                "the side-by-sides train at the normal rate. Use adamw, adamw8bit or muon for the boost."
            )

    def _micro_step(self, index: int) -> torch.Tensor:
        if self.samples[index][0] in self.composites:
            self._step_boost = self.lr_boost
        return super()._micro_step(index)

    def _optimizer_step(self, network: Any) -> None:
        boost, self._step_boost = self._step_boost, 1.0
        if boost == 1.0 or not self._boostable:
            super()._optimizer_step(network)
            return
        groups = self.optimizer.param_groups
        before = [group["lr"] for group in groups]
        for group in groups:
            group["lr"] *= boost
        try:
            super()._optimizer_step(network)
        finally:
            if self.lr_scheduler is None:  # a scheduler sets the rate afresh each step
                for group, lr in zip(groups, before, strict=True):
                    group["lr"] = lr


def run_lora() -> None:
    """train_krea2_lora.py's entry point: the command line, then the LoRA (masks and subjects when the config asks)."""
    args = training_arguments("Train a Krea 2 LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    Krea2SubjectsLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()
