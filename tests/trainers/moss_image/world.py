"""A tiny MOSS-Image training world under one folder: images, a layout checkpoint and a config."""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from tests.maltrainer.models.moss_image.tiny import json_tensor, tiny_model

PREFIX = "diffusion_model."
BASE_CONFIG: dict[str, Any] = {
    "dataset": {
        "resolution": 32,
        "aspect_mode": "bucket",
        "trigger": "zed",
        "default_caption": "a photo of zed",
        "cache": False,
    },
    "finetune": {"trainable_prefixes": ["layers."], "linears_only": True},
    "prior": {"weight": 0.0, "classes": {"zed": "person"}},
    "train": {
        "max_steps": 3,
        "save_state": True,
        "gradient_checkpointing": False,
        "learning_rate": 1e-3,
        "automagic_min_lr": 1e-3,
        "automagic_max_lr": 1e-3,
        "monitor_every": 2,
        "timestep_uniform_mix": 0.0,
    },
    "dtype": "bfloat16",
    "mixed_precision": "no",
    "seed": 42,
    "output": {"name": "ft", "dtype": "bfloat16"},
}


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def write_layout_checkpoint(path: Path, seed: int = 0) -> Path:
    """The tiny model as a ComfyUI bf16 file: prefixed keys, plus the entries training does not use."""
    tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in tiny_model(seed=seed).state_dict().items()}
    tensors["rope.bands"] = torch.arange(4, dtype=torch.float32)
    tensors["layers.0.comfy_attention.config"] = json_tensor({"attention": "sdpa"})
    save_file({PREFIX + key: value for key, value in tensors.items()}, str(path))
    return path


def write_image(path: Path, caption: str | None = None, size: tuple[int, int] = (32, 32), shade: int = 0) -> Path:
    """An image whose colour follows its name (and ``shade``), with a ``.txt`` caption when one is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(path.name.encode("utf-8")) + shade
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
    def layout(self) -> Path:
        return self.root / "base.safetensors"

    def add_images(self, folder: str | Path, count: int, caption: str | None = "a photo of zed") -> list[Path]:
        target = folder if isinstance(folder, Path) else self.data / folder
        return [write_image(target / f"img{i}.png", caption) for i in range(count)]

    def config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """The base config pointing everything at this folder, with ``base`` and then ``sections`` merged over it."""
        paths = {
            "model": {
                "transformer_path": str(self.layout),
                "vae_path": str(self.root / "vae.safetensors"),
                "text_encoder_path": str(self.root / "text.safetensors"),
            },
            "dataset": {"path": str(self.data)},
            "output": {"path": str(self.out)},
        }
        return merge(merge(merge(BASE_CONFIG, paths), base or {}), sections)

    def lora_config(self, base: dict[str, Any] | None = None, **sections: Any) -> dict[str, Any]:
        """A LoRA run: the same world without the finetune-only sections, rank 4, saved as moss_image_lora."""
        config = self.config(base, **sections)
        for finetune_only in ("finetune", "prior"):
            config.pop(finetune_only)
        del config["output"]["name"]
        config["train"].setdefault("automagic_fused", False)
        return merge({"lora": {"rank": 4, "alpha": 4}}, config)


def named_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def same_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def base_weights(ws: Workspace) -> dict[str, torch.Tensor]:
    """The layout checkpoint's module weights, as the trainer first loads them."""
    stored = load_file(str(ws.layout))
    return {k[len(PREFIX) :]: v for k, v in stored.items() if k[len(PREFIX) :] in tiny_model().state_dict()}


SAMPLE: dict[str, Any] = {
    "every": 2,
    "at_start": True,
    "rows": ["raw", "ema"],
    "prompts": ["a photo of zed", {"prompt": "a photo of a person", "enabled": False}],
    "steps": 2,
    "cfg": 3.0,
    "width": 32,
    "height": 32,
    "seed": 7,
}
