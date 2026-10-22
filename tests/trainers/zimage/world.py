"""A tiny Z-Image training world under one folder: images, and the configs that point the trainers at it."""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

BASE_CONFIG: dict[str, Any] = {
    "model": {"name_or_path": "tiny/zimage"},
    "dataset": {"resolution": 16, "trigger": "zed", "default_caption": "a photo"},
    "train": {"num_epochs": 1, "batch_size": 1, "save_every": 100, "learning_rate": 1e-3},
    "dtype": "float32",
    "mixed_precision": "no",
    "seed": 42,
}


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def write_image(path: Path, caption: str | None = None, size: tuple[int, int] = (16, 16)) -> Path:
    """An image whose colour follows its name, with a ``.txt`` caption when one is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(path.name.encode("utf-8"))
    Image.new("RGB", size, (seed % 256, seed // 256 % 256, seed // 65536 % 256)).save(path)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path


@dataclass
class Workspace:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def out(self) -> Path:
        return self.root / "out"

    def add_images(self, count: int, caption: str | None = "a photo of person") -> list[Path]:
        return [write_image(self.data / f"img{i}.png", caption) for i in range(count)]

    def config(self, **sections: Any) -> dict[str, Any]:
        """The base config pointing at this folder, with ``sections`` merged over it (``key=None`` removes it)."""
        paths = {"dataset": {"path": str(self.data)}, "output": {"path": str(self.out)}}
        merged = merge(merge(BASE_CONFIG, paths), sections)
        return {key: value for key, value in merged.items() if value is not None}

    def lora_config(self, **sections: Any) -> dict[str, Any]:
        return self.config(**merge({"lora": {"rank": 4, "alpha": 4}}, sections))

    def embedding_config(self, **sections: Any) -> dict[str, Any]:
        defaults = {"embedding": {"trigger": "sks", "num_vectors": 1, "init_text": "a photo"}}
        return self.config(**merge(defaults, sections))


def named_weights(model: Any) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
    """The names whose tensors differ."""
    return {name for name in before if not torch.equal(before[name], after[name])}


def lora_layers(trainer: Any) -> list[Any]:
    """The trainer's LoRA layers (``ModuleDict`` values are plain ``Module`` to mypy)."""
    return list(trainer.lora_network.lora_layers.values())
