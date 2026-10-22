"""A tiny Qwen-Image-2.1 training world under one folder: images, a config, and helpers over the results."""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

BASE_CONFIG: dict[str, Any] = {
    "dataset": {"resolution": 32, "trigger": "zed", "default_caption": "a photo of zed"},
    "lora": {"rank": 4, "alpha": 4},
    "train": {"max_steps": 3, "save_every": 100, "optimizer": "adamw", "learning_rate": 1e-3},
    "dtype": "float32",
    "mixed_precision": "no",
    "seed": 42,
    "output": {"dtype": "bfloat16"},
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
    """A solid image whose colour follows its name, with a ``.txt`` caption when one is given."""
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

    def add_images(self, count: int, caption: str | None = "a photo of zed", folder: str = "p") -> list[Path]:
        return [write_image(self.data / folder / f"img{i}.png", caption) for i in range(count)]

    def config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """The base config pointing everything at this folder, with ``base`` and then ``sections`` merged over it."""
        paths = {
            "model": {
                "transformer_path": str(self.root / "transformer.safetensors"),
                "vae_path": str(self.root / "vae.safetensors"),
                "text_encoder_path": str(self.root / "text.safetensors"),
            },
            "dataset": {"path": str(self.data)},
            "output": {"path": str(self.out)},
        }
        return merge(merge(merge(BASE_CONFIG, paths), base or {}), sections)


def named_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def same_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
