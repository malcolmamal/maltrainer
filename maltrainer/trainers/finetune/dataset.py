"""A full finetune's data, for any image trainer on ``ImageLoRATrainer``: regularisation images beside the subject
images, per-folder repeats, the class captions prior preservation records, the dataset fingerprint a resume checks,
and --suggest-steps for all of them. Mixed in ahead of the trainer's previews (the MOSS-Image and Krea 2 finetunes),
whose ``_extra_prompts`` it extends.

    dataset:
      repeats: 1
      folder_repeats: {people/ben: 3}   # maltrainer/finetune/repeats.py
      reg_path: "datasets/reg"          # regularisation images, at train.reg_lr_multiplier x the step
      reg_repeats: 1
    prior: {weight: 1.0, classes: {ben: man}}   # maltrainer/finetune/prior.py
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from maltrainer.data.minimax_dataset import MediaSource, scan_media_sources
from maltrainer.finetune.classes import class_caption
from maltrainer.finetune.notes import step_notes
from maltrainer.finetune.prior import PriorPreservation, resolve_classes
from maltrainer.finetune.repeats import folder_repeats, repeat_order, samples_per_epoch
from maltrainer.training.advice import StepAdvice
from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.utils.gallery import Gallery
from maltrainer.utils.previews import Previews
from maltrainer.utils.step_planner import Group, folder_counts

logger = logging.getLogger(__name__)


class FinetuneDataset(StepAdvice, ImageLoRATrainer):
    config: dict[str, Any]
    samples: list[tuple[str, str]]
    previews: Previews
    reg_keys: set[str]  # dataset.reg_path: regularisation images, trained at train.reg_lr_multiplier x the step
    reg_multiplier: float
    prior: PriorPreservation | None
    prior_classes: dict[str, str]
    prior_path: Path | None
    gallery: Gallery

    def _init_finetune_data(self, config: dict[str, Any]) -> None:
        """Regularisation, prior preservation (prior.weight > 0) and the gallery (key g) from the config."""
        self.reg_keys = set()
        self.reg_multiplier = float(config.get("train", {}).get("reg_lr_multiplier", 0.2))
        prior_config = config.get("prior", {}) or {}
        self.prior = None
        self.prior_classes = {}
        self.prior_path = None
        if float(prior_config.get("weight", 0.0) or 0.0) > 0:
            self.prior_classes = resolve_classes(prior_config, config.get("dataset", {}).get("trigger"))
            self.prior = PriorPreservation(
                weight=float(prior_config["weight"]),
                samples_per_image=int(prior_config.get("samples_per_image", 8)),
                every=int(prior_config.get("every", 1)),
                seed=int(config.get("seed", 42)),
            )
        self.gallery = Gallery.from_config(config, self.prior_classes)  # key g (maltrainer/utils/gallery.py)

    def _sources(self) -> list[MediaSource]:
        sources = super()._sources()
        dataset_config = self.config["dataset"]
        reg_path = dataset_config.get("reg_path")
        if not reg_path:
            return sources
        reg = scan_media_sources(
            reg_path, kinds=["image"], default_caption=dataset_config.get("reg_default_caption", "")
        )
        if not reg:
            raise ValueError(f"dataset.reg_path {reg_path} has no images")
        self.reg_keys = {str(source.path) for source in reg}
        logger.info("Regularisation: %d images from %s at x%g the step", len(reg), reg_path, self.reg_multiplier)
        return sources + reg

    @classmethod
    def step_groups(cls, config: dict[str, Any]) -> list[Group]:
        groups = super().step_groups(config)
        dataset_config = config["dataset"]
        groups[0].folder_repeats = dict(dataset_config.get("folder_repeats") or {})
        reg_path = dataset_config.get("reg_path")
        if reg_path:
            groups.append(
                Group(
                    "Regularisation",
                    reg_path,
                    max(float(dataset_config.get("reg_repeats", 1)), 0),
                    folder_counts(reg_path),
                )
            )
        return groups

    @classmethod
    def step_notes(cls, config: dict[str, Any]) -> list[str]:
        """Folders prior preservation would miss (see ``maltrainer.finetune.notes``)."""
        return step_notes(config)

    def _dry_run_samples(self) -> int:
        """One epoch, counted from the folders as --suggest-steps does (per-folder repeats included)."""
        return sum(group.samples for group in self.step_groups(self.config))

    def _rel(self, key: str) -> str:
        """A sample's name relative to its dataset folder, so a moved dataset still matches."""
        root = self.config["dataset"].get("reg_path") if key in self.reg_keys else self.config["dataset"]["path"]
        return Path(os.path.relpath(key, root)).as_posix()

    def _prior_captions(self) -> dict[str, Any]:
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

    def _extra_prompts(self) -> list[str]:
        gallery = self.gallery.captions
        if gallery and self.previews.cfg != 1.0:
            gallery = [*gallery, self.previews.negative]
        return super()._extra_prompts() + sorted(set(self._prior_captions().values())) + gallery

    def _repeat_groups(self) -> list[tuple[Any, ...]]:
        """(sample indices, repeats) per subject folder (dataset.folder_repeats, else
        dataset.repeats), and the regularisation images with dataset.reg_repeats."""
        dataset_config = self.config["dataset"]
        default = float(dataset_config.get("repeats", 1) or 1)
        table = dataset_config.get("folder_repeats") or {}
        folders: dict[str, list[int]] = {}
        reg = []
        for i, (key, _) in enumerate(self.samples):
            if key in self.reg_keys:
                reg.append(i)
            else:
                folders.setdefault(
                    Path(os.path.relpath(Path(key).parent, dataset_config["path"])).as_posix(), []
                ).append(i)
        groups = [(indices, folder_repeats(folder, table, default)) for folder, indices in sorted(folders.items())]
        return [*groups, (reg, float(max(dataset_config.get("reg_repeats", 1), 0)))]

    def _epoch_order(self) -> list[int]:
        return repeat_order(self._repeat_groups())

    def _epoch_size(self) -> int:
        """One epoch's samples, without drawing its order (so without touching Python's random)."""
        return sum(samples_per_epoch(len(indices), repeats) for indices, repeats in self._repeat_groups())

    def _dataset_fingerprint(self) -> str:
        """The samples (paths relative to the dataset folder, captions) and repeats, so a moved
        dataset still matches and any added, removed or recaptioned image does not."""
        root = self.config["dataset"]["path"]
        digest = hashlib.sha1(str(self.config["dataset"].get("repeats", 1)).encode("utf-8"), usedforsecurity=False)
        if self.config["dataset"].get("folder_repeats"):  # only then, so earlier states keep their fingerprint
            digest.update(repr(sorted(self.config["dataset"]["folder_repeats"].items())).encode("utf-8"))
        for key, caption in self.samples:
            if key in self.reg_keys:  # marked only then, so states without regularisation keep their fingerprint
                digest.update(f"\x1ereg:{self._rel(key)}\x1f{caption}".encode())
            else:
                digest.update(f"\x1e{Path(os.path.relpath(key, root)).as_posix()}\x1f{caption}".encode())
        return digest.hexdigest()

    def precache_dataset(self) -> None:
        Path(self.config["model"]["transformer_path"]).stat()  # a typo fails here, not after all the caching
        super().precache_dataset()
