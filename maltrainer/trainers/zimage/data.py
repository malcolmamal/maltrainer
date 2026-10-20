"""The Z-Image LoRA trainer's data: the image dataset with its default captions, its loader, and the file count
the dry run plans from."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from maltrainer.data.dataset import ZImageDataset
from maltrainer.data.walk import count_files

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_loader(config: dict[str, Any], trigger: str | None) -> tuple[ZImageDataset, DataLoader[Any]]:
    """The images with their captions (the ``trigger`` in front) and a shuffling loader."""
    dataset_config = config["dataset"]
    dataset = ZImageDataset(
        data_root=dataset_config["path"],
        resolution=dataset_config.get("resolution", 512),
        center_crop=dataset_config.get("center_crop", True),
        random_flip=dataset_config.get("random_flip", False),
        trigger_token=trigger,
        repeats=dataset_config.get("repeats", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["train"].get("batch_size", 1),
        shuffle=True,
        num_workers=dataset_config.get("num_workers", 0),
    )
    return dataset, loader


def make_loader(config: dict[str, Any]) -> tuple[ZImageDataset, DataLoader[Any]]:
    """The LoRA trainer's dataset: ``dataset.default_caption`` for captions that are empty or just the trigger,
    and a warning for empty captions without a trigger."""
    dataset_config = config["dataset"]
    dataset, loader = image_loader(config, dataset_config.get("trigger", None))
    default_caption = dataset_config.get("default_caption", "")
    if default_caption:
        dataset.captions = with_default_captions(
            dataset.captions, (dataset_config.get("trigger") or "").strip(), default_caption
        )
    empty_captions = sum(1 for c in dataset.captions if not c)
    if not dataset_config.get("trigger", None) and empty_captions > 0:
        logger.warning(
            "Detected %d empty captions and no trigger token. "
            "Training may fail to learn a consistent concept. "
            "Add dataset.trigger or provide caption .txt files.",
            empty_captions,
        )
    return dataset, loader


def with_default_captions(captions: list[str], trigger: str, default_caption: str) -> list[str]:
    """An empty caption, or one that is just the trigger (the dataset prepends it), becomes the default caption,
    with the trigger in front when the default does not name it."""
    defaulted = []
    for c in captions:
        c_clean = c.strip()
        if not c_clean or c_clean == trigger:
            if trigger and default_caption and trigger not in default_caption:
                defaulted.append(f"{trigger} {default_caption}")
            else:
                defaulted.append(default_caption)
        else:
            defaulted.append(c)
    return defaulted


def count_dataset_samples(dataset_config: dict[str, Any]) -> int:
    data_root = Path(str(dataset_config.get("path", "")))
    if not data_root.exists():
        logger.warning("Dataset path does not exist: %s", data_root)
        return 1
    sample_count = count_files(data_root, IMAGE_EXTENSIONS)
    repeats = int(dataset_config.get("repeats", 1) or 1)
    return max(sample_count, 1) * max(repeats, 1)
