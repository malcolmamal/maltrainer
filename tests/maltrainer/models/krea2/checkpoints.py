"""Synthetic ComfyUI-layout Krea 2 checkpoints written to disk for the loader tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from maltrainer.models.comfy_fp8 import FP8_MAX
from maltrainer.models.krea2.loading import is_block_linear
from tests.maltrainer.models.krea2.tiny_model import tiny_transformer

BLOCK_LINEARS = 2 * 8


def dense_state(dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(dtype) for k, v in tiny_transformer().state_dict().items()}


def fp8_state() -> Dict[str, torch.Tensor]:
    """Block Linears as fp8 with one scale per tensor, the rest bf16, as an fp8_scaled file stores them."""
    state = dense_state(torch.bfloat16)
    for key in [k for k in state if is_block_linear(k)]:
        weight = dense_state()[key]
        scale = (weight.abs().amax() / FP8_MAX).float()
        state[key] = (weight / scale).to(torch.float8_e4m3fn)
        state[key.replace(".weight", ".weight_scale")] = scale
    return state


def write(path: Path, state: Dict[str, torch.Tensor], prefix: str = "", fp8_layers: Optional[list] = None) -> Path:
    metadata = None
    if fp8_layers is not None:
        layers = {prefix + name: {"format": "float8_e4m3fn"} for name in fp8_layers}
        metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
    save_file({prefix + k: v.contiguous() for k, v in state.items()}, str(path), metadata=metadata)
    return path


def fp8_layer_names(state: Dict[str, torch.Tensor]) -> list:
    return [k[: -len(".weight")] for k in state if is_block_linear(k)]
