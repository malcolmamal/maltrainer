"""A tiny Fibo 2.0 checkpoint in Comfy-Org's INT8 ConvRot layout: the main Linears as int8 codes with a per-row scale
and a marker, everything else dense, a kernel hint per attention."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from maltrainer.models.fibo2.model import Fibo2Transformer
from maltrainer.models.minimax_h3.int8_rotation import _quantize_rows, rotate_activation

from .tiny_model import tiny_transformer

GROUP = 16
QUANTIZED = (".attention.qkv", ".attention.out", ".feed_forward.w13", ".feed_forward.w2")
MARKER = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP})


def _bytes(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode()), dtype=torch.uint8)


def int8_state(model: Fibo2Transformer | None = None) -> tuple[dict[str, torch.Tensor], list[str]]:
    """(the file's tensors, the names of the Linears stored as int8)."""
    source = (model or tiny_transformer()).state_dict()
    tensors: dict[str, torch.Tensor] = {}
    stored: list[str] = []
    for key, value in source.items():
        base = key.removesuffix(".weight")
        if key.endswith(".weight") and base.endswith(QUANTIZED) and base.startswith(("layers.", "noise_refiner.")):
            codes, scale = _quantize_rows(rotate_activation(value, GROUP))
            tensors[key], tensors[f"{base}.weight_scale"], tensors[f"{base}.comfy_quant"] = codes, scale, _bytes(MARKER)
            stored.append(base)
        else:
            tensors[key] = value
    tensors["layers.0.attention.comfy_attention.config"] = _bytes('{"attention": "comfy_kitchen_int8"}')
    return tensors, stored


def write_int8(path: Path, model: Fibo2Transformer | None = None) -> tuple[Path, list[str]]:
    tensors, stored = int8_state(model)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    return path, stored
