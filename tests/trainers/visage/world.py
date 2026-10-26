"""A tiny Visage-Image training world under one folder: images, a plain checkpoint and a config."""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from tests.maltrainer.models.visage.tiny_model import tiny_transformer

BASE_CONFIG: dict[str, Any] = {
    "dataset": {"resolution": 32, "trigger": "zed", "default_caption": "a photo of zed"},
    "lora": {"rank": 4, "alpha": 4, "dtype": "float32"},
    "train": {
        "max_steps": 3,
        "save_every": 0,
        "learning_rate": 1e-3,
        "automagic_min_lr": 1e-3,
        "automagic_max_lr": 1e-3,
        "automagic_polarity_history": 2,
    },
    "dtype": "float32",
    "mixed_precision": "no",
    "seed": 42,
    "output": {"name": "visage", "dtype": "bfloat16"},
}


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def write_image(path: Path, caption: str | None = None, size: tuple[int, int] = (32, 32)) -> Path:
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

    @property
    def checkpoint(self) -> Path:
        return self.root / "visage.safetensors"

    def add_images(self, count: int, caption: str | None = "a photo of zed", size: tuple[int, int] = (32, 32)) -> None:
        for i in range(count):
            write_image(self.data / f"img{i}.png", caption, size)

    def config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """The base config pointing everything at this folder, with ``base`` and then ``sections`` merged over it."""
        paths = {
            "model": {
                "transformer_path": str(self.checkpoint),
                "vae_path": str(self.root / "vae.safetensors"),
                "text_encoder_path": str(self.root / "text.safetensors"),
            },
            "dataset": {"path": str(self.data)},
            "output": {"path": str(self.out)},
        }
        return merge(merge(merge(BASE_CONFIG, paths), base or {}), sections)


def write_checkpoint(path: Path) -> Path:
    """The tiny DiT as a plain float32 file under the module names of the Hub repo."""
    save_file(
        {key: value.detach().clone().contiguous() for key, value in tiny_transformer().state_dict().items()}, str(path)
    )
    return path


def named_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def same_values(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def base_weights(ws: Workspace) -> dict[str, torch.Tensor]:
    """The checkpoint's weights, which are the module names already."""
    return load_file(str(ws.checkpoint))
