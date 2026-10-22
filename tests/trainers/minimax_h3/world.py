"""A tiny MiniMax H3 training world under one folder: media, a config, and helpers over the results."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .media import sound, write_clip, write_image, write_wav

BASE_CONFIG: dict[str, Any] = {
    "dataset": {
        "resolution": 32,
        "aspect_mode": "bucket",
        "max_frames": 22,
        "trigger": "zed",
        "default_caption": "a photo of zed",
        "audio_placeholder_resolution": 32,
    },
    "lora": {"rank": 4, "alpha": 4},
    "train": {
        "num_epochs": 1,
        "save_every": 100,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "timestep_distribution": "krea2_shift",
    },
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


@dataclass
class Workspace:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def out(self) -> Path:
        return self.root / "out"

    def add_images(self, count: int, caption: str | None = "a photo of zed", folder: str = "stills") -> list[Path]:
        return [write_image(self.data / folder / f"img{i}.png", caption) for i in range(count)]

    def add_clip(
        self, name: str = "clip", frames: int = 5, seconds: float | None = 0.3, caption: str | None = None
    ) -> Path:
        audio = None if seconds is None else sound(seconds)
        return write_clip(self.data / "clips" / f"{name}.avi", frames, audio, caption)

    def add_voice(self, name: str = "voice", seconds: float = 1.0, caption: str | None = None) -> Path:
        return write_wav(self.data / "voices" / f"{name}.wav", sound(seconds), caption=caption)

    def add_silence(self, name: str = "silent", seconds: float = 1.0) -> Path:
        return write_wav(self.data / "voices" / f"{name}.wav", np.zeros((2, round(seconds * 16000)), np.float32))

    def config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """The base config pointing everything at this folder, with ``base`` and then ``sections`` merged over it."""
        paths = {
            "model": {
                "transformer_path": str(self.root / "transformer.safetensors"),
                "vae_path": str(self.root / "vae.safetensors"),
                "audio_vae_path": str(self.root / "audio_vae.safetensors"),
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
