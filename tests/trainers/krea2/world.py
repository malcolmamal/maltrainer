"""A tiny Krea 2 training world under one folder: images, a bf16 checkpoint and the two configs."""

from __future__ import annotations

import copy
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from maltrainer.models.comfy_fp8 import quantize_rows
from maltrainer.models.krea2.loading import is_block_linear

from .tiny import tiny_model

PREFIX = "model.diffusion_model."
BASE_CONFIG: dict[str, Any] = {
    "dataset": {"resolution": 32, "aspect_mode": "bucket", "trigger": "zed", "default_caption": "a photo of zed"},
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
    "output": {"name": "k2"},
}
LORA: dict[str, Any] = {"lora": {"rank": 4, "alpha": 4, "dtype": "float32"}, "output": {"dtype": "bfloat16"}}
FINETUNE: dict[str, Any] = {
    "finetune": {"rotate_every": 2},
    "train": {"monitor_every": 2, "save_state": True},
    "dtype": "bfloat16",
    "output": {"name": "ft"},
}


def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def _text_linear(key: str) -> bool:
    return key.startswith("txtfusion.") and key.endswith(".weight") and (".attn." in key or ".mlp." in key)


def write_checkpoint(path: Path, prefix: str = PREFIX, fp8: bool = False, fp8_text: bool = False) -> Path:
    """The tiny DiT as a bf16 ComfyUI file, or with fp8 block Linears and their per-row scales; ``fp8_text`` also
    stores the text fusion's Linears in fp8 with one scale each, as ComfyUI's krea2_raw_fp8_scaled does."""
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, str]] = {}
    for key, value in tiny_model().state_dict().items():
        if (fp8 and is_block_linear(key)) or (fp8_text and _text_linear(key)):
            if is_block_linear(key):
                tensors[prefix + key], scale = quantize_rows(value)
            else:
                scale = (value.abs().max() / 448.0).float()
                tensors[prefix + key] = (value / scale).to(torch.float8_e4m3fn)
            tensors[prefix + key[: -len(".weight")] + ".weight_scale"] = scale
            layers[prefix + key[: -len(".weight")]] = {"format": "float8_e4m3fn"}
        else:
            tensors[prefix + key] = value.to(torch.bfloat16).contiguous()
    metadata = {"_quantization_metadata": json.dumps({"layers": layers})} if layers else None
    save_file(tensors, str(path), metadata=metadata)
    return path


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
        return self.root / "krea2.safetensors"

    def add_images(self, count: int, caption: str | None = "a photo of zed", size: tuple[int, int] = (32, 32)) -> None:
        for i in range(count):
            write_image(self.data / f"img{i}.png", caption, size)

    def _paths(self) -> dict[str, Any]:
        return {
            "model": {
                "transformer_path": str(self.checkpoint),
                "vae_path": str(self.root / "vae.safetensors"),
                "text_encoder_path": str(self.root / "text.safetensors"),
            },
            "dataset": {"path": str(self.data)},
            "output": {"path": str(self.out)},
        }

    def lora_config(self, **sections: Any) -> dict[str, Any]:
        """A LoRA run pointing everything at this folder, with ``sections`` merged over it."""
        return merge(merge(merge(BASE_CONFIG, self._paths()), LORA), sections)

    def finetune_config(self, **sections: Any) -> dict[str, Any]:
        """A finetune run (two-step windows, bf16) pointing everything at this folder."""
        return merge(merge(merge(BASE_CONFIG, self._paths()), FINETUNE), sections)


def named_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def same_values(left: dict[Any, Any], right: dict[Any, Any]) -> bool:
    """Equal keys and, tensor by tensor, bit-identical values."""
    if left.keys() != right.keys():
        return False
    return all(
        torch.equal(left[key], right[key]) if torch.is_tensor(left[key]) else left[key] == right[key] for key in left
    )


def stored_weights(path: Path) -> dict[str, torch.Tensor]:
    """A checkpoint's tensors under the module names (the container prefix stripped)."""
    return {key.removeprefix(PREFIX): value for key, value in load_file(str(path)).items()}
