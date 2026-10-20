"""
Runtime support for Comfy NVFP4/AWQ quantized text encoder checkpoints.

The quantized layers live in ``comfy_quant_layers``; every name is re-exported here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from maltrainer.models.minimax_h3.comfy_quant_layers import (
    _DEQUANT_CHUNK_ELEMENTS,
    _NVFP4_LUT_CACHE,
    NVFP4_BLOCK_SIZE,
    ComfyInt8Embedding,
    ComfyNvfp4Linear,
    get_nvfp4_lut,
    unswizzle_nvfp4_scales,
)

logger = logging.getLogger(__name__)

COMFY_QUANT_SUFFIX = ".comfy_quant"

__all__ = [
    "COMFY_QUANT_SUFFIX",
    "NVFP4_BLOCK_SIZE",
    "_DEQUANT_CHUNK_ELEMENTS",
    "_NVFP4_LUT_CACHE",
    "ComfyInt8Embedding",
    "ComfyNvfp4Linear",
    "_parse_quant_config",
    "_replace_submodule",
    "get_nvfp4_lut",
    "has_comfy_quantized_layers",
    "load_comfy_quantized_state_dict",
    "unswizzle_nvfp4_scales",
]


def has_comfy_quantized_layers(path: str | Path) -> bool:
    """Return whether a safetensors checkpoint contains Comfy quantization markers."""
    path = Path(path)
    if not path.is_file():
        return False
    with safe_open(path, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        return any(key.endswith(COMFY_QUANT_SUFFIX) for key in source_keys)


def _parse_quant_config(blob: torch.Tensor) -> dict[str, object]:
    try:
        value = json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Comfy quantization marker") from exc
    if not isinstance(value, dict):
        raise TypeError("Comfy quantization marker must contain a JSON object")
    return value


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


def _replacement_for(
    module: nn.Module,
    prefix: str,
    quant_format: str,
    tensors: tuple[torch.Tensor, torch.Tensor],
    handle: Any,
    source_keys: list[str],
    output_dtype: torch.dtype,
) -> tuple[nn.Module, set[str]]:
    """The quantized module that replaces ``module`` and the extra checkpoint keys it consumes."""
    weight, weight_scale = tensors
    if isinstance(module, nn.Embedding):
        if quant_format != "int8_tensorwise":
            raise ValueError(f"Unsupported Comfy embedding quantization {quant_format!r} on {prefix}")
        return ComfyInt8Embedding(weight, weight_scale, output_dtype), set()
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Comfy quant marker {prefix} maps to {type(module).__name__}; expected Linear or Embedding")
    if quant_format != "nvfp4":
        raise ValueError(f"Unsupported Comfy Linear quantization {quant_format!r} on {prefix}")
    scale_2_key = f"{prefix}.weight_scale_2"
    if scale_2_key not in source_keys:
        raise ValueError(f"{prefix} is missing weight_scale_2")
    pre_scale_key = f"{prefix}.pre_quant_scale"
    replacement = ComfyNvfp4Linear(
        module,
        weight,
        weight_scale,
        handle.get_tensor(scale_2_key),
        handle.get_tensor(pre_scale_key) if pre_scale_key in source_keys else None,
        output_dtype,
    )
    consumed_keys = {scale_2_key}
    if pre_scale_key in source_keys:
        consumed_keys.add(pre_scale_key)
    return replacement, consumed_keys


def _attach_marker(
    model: nn.Module,
    handle: Any,
    source_keys: list[str],
    marker_key: str,
    key_map: Callable[[str], str],
    output_dtype: torch.dtype,
) -> tuple[str, set[str]]:
    """Swap in the quantized module one marker describes; returns its format and the keys it consumed."""
    prefix = marker_key[: -len(COMFY_QUANT_SUFFIX)]
    config = _parse_quant_config(handle.get_tensor(marker_key))
    quant_format = config.get("format")
    if not isinstance(quant_format, str):
        raise TypeError(f"{prefix} has no quantization format")
    module_path = key_map(prefix)
    module = model.get_submodule(module_path)
    weight_key = f"{prefix}.weight"
    scale_key = f"{prefix}.weight_scale"
    if weight_key not in source_keys or scale_key not in source_keys:
        raise ValueError(f"{prefix} is missing its quantized weight or scale")
    tensors = (handle.get_tensor(weight_key), handle.get_tensor(scale_key))
    replacement, extra_keys = _replacement_for(module, prefix, quant_format, tensors, handle, source_keys, output_dtype)
    _replace_submodule(model, module_path, replacement)
    return quant_format, {marker_key, weight_key, scale_key, *extra_keys}


def _plain_tensors(
    handle: Any,
    source_keys: list[str],
    consumed: set[str],
    expected: set[str],
    key_map: Callable[[str], str],
    checkpoint: Path,
) -> dict[str, torch.Tensor]:
    """The checkpoint tensors that stay plain, under the model's key names."""
    state_dict: dict[str, torch.Tensor] = {}
    for source_key in source_keys:
        if source_key in consumed:
            continue
        source_prefix, separator, suffix = source_key.rpartition(".")
        if not separator:
            raise ValueError(f"{checkpoint.name} contains unexpected key {source_key!r}")
        target_key = f"{key_map(source_prefix)}.{suffix}"
        if target_key not in expected:
            raise ValueError(f"{checkpoint.name} contains unexpected key {source_key!r}")
        state_dict[target_key] = handle.get_tensor(source_key)
    return state_dict


@torch.no_grad()
def load_comfy_quantized_state_dict(
    model: nn.Module,
    checkpoint: Path | str,
    *,
    key_map: Callable[[str], str],
    output_dtype: torch.dtype,
) -> int:
    """Load a Comfy NVFP4/INT8 safetensors checkpoint into a model."""
    checkpoint = Path(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        marker_keys = [key for key in source_keys if key.endswith(COMFY_QUANT_SUFFIX)]
        if not marker_keys:
            raise ValueError(f"{checkpoint.name} does not contain Comfy quantized layers")

        consumed: set[str] = set()
        format_counts: dict[str, int] = {}
        for marker_key in marker_keys:
            quant_format, consumed_keys = _attach_marker(model, handle, source_keys, marker_key, key_map, output_dtype)
            format_counts[quant_format] = format_counts.get(quant_format, 0) + 1
            consumed.update(consumed_keys)

        expected = set(model.state_dict())
        state_dict = _plain_tensors(handle, source_keys, consumed, expected, key_map, checkpoint)

    missing = sorted(expected - set(state_dict))
    if missing:
        raise ValueError(f"{checkpoint.name} is missing {len(missing)} tensor(s), examples: {missing[:5]}")
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"Strict NVFP4/AWQ load failed: {info}")
    logger.info("Attached %d Comfy-quantized H3 layers: %s", len(marker_keys), format_counts)
    return len(marker_keys)
