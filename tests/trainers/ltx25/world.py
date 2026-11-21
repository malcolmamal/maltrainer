"""A tiny LTX-2.5 training world under one folder: stills, clips with sound, a plain checkpoint and a config."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from tests.maltrainer.models.ltx25.tiny_model import tiny_transformer
from tests.trainers.minimax_h3.media import sound, write_clip, write_image

HEADS = {
    "positional_embedding_max_pos": [20, 64, 64],
    "num_attention_heads": 2,
    "audio_num_attention_heads": 2,
    "connector_attention_head_dim": 48,
    "audio_connector_attention_head_dim": 24,
}
BASE_CONFIG: dict[str, Any] = {
    "dataset": {"resolution": 32, "fps": 24, "clip_frames": 9, "trigger": "zed", "default_caption": "a photo of zed"},
    "lora": {"rank": 4, "alpha": 4, "dtype": "float32"},
    "train": {
        "max_steps": 3,
        "save_every": 0,
        "gradient_checkpointing": False,
        "learning_rate": 1e-3,
        "automagic_min_lr": 1e-3,
        "automagic_max_lr": 1e-3,
        "automagic_polarity_history": 2,
    },
    "dtype": "float32",
    "mixed_precision": "no",
    "seed": 42,
    "output": {"name": "ltx25", "dtype": "bfloat16"},
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

    @property
    def checkpoint(self) -> Path:
        return self.root / "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    def add_images(self, count: int, caption: str | None = "a photo of zed", size: tuple[int, int] = (32, 32)) -> None:
        for i in range(count):
            write_image(self.data / f"img{i}.png", caption, size)

    def add_clip(
        self, name: str = "clip0", frames: int = 9, seconds: float | None = 0.5, caption: str | None = "zed walks"
    ) -> Path:
        """A 24 fps clip of ``frames`` frames, with ``seconds`` of 16 kHz stereo sound (none if ``seconds`` is None)."""
        audio = None if seconds is None else sound(seconds, seed=len(name))
        return write_clip(self.data / f"{name}.avi", frames, audio, caption)

    def config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """The base config pointing everything at this folder, with ``base`` and then ``sections`` merged over it."""
        paths = {
            "model": {
                "transformer_path": str(self.checkpoint),
                "video_vae_path": str(self.root / "video_vae.safetensors"),
                "audio_vae_path": str(self.root / "audio_vae.safetensors"),
                "text_encoder_path": str(self.root / "text.safetensors"),
            },
            "dataset": {"path": str(self.data)},
            "output": {"path": str(self.out)},
        }
        return merge(merge(merge(BASE_CONFIG, paths), base or {}), sections)


def write_checkpoint(path: Path) -> Path:
    """The tiny DiT (audio stream included) as a plain float32 file in ComfyUI's layout, with its config metadata."""
    state = {key: value.detach().clone().contiguous() for key, value in tiny_transformer().state_dict().items()}
    save_file(state, str(path), metadata={"config": json.dumps({"transformer": HEADS})})
    return path


def named_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def same_values(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def base_weights(ws: Workspace) -> dict[str, torch.Tensor]:
    """The checkpoint's weights, which are the module names already."""
    return load_file(str(ws.checkpoint))
