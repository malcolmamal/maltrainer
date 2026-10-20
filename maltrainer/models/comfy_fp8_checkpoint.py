"""
Reading ComfyUI "fp8_scaled" checkpoints: the three quantization marker styles, and the container prefix.

See ``comfy_fp8`` for what the checkpoints are and how they are trained on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

FP8_FORMATS = {"float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}


def _layer_configs(handle: Any, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """prefix -> per-layer quant config, from whichever marker style the file uses."""
    metadata = handle.metadata() or {}
    if "_quantization_metadata" in metadata:
        return dict(json.loads(metadata["_quantization_metadata"]).get("layers", {}))

    configs: dict[str, dict[str, Any]] = {}
    for key in keys:
        if key.endswith(".comfy_quant"):
            blob = handle.get_tensor(key)
            configs[key[: -len(".comfy_quant")]] = json.loads(bytes(blob.tolist()).decode("utf-8"))
    if configs:
        return configs

    legacy = [key for key in keys if key == "scaled_fp8" or key.endswith(".scaled_fp8")]
    if legacy:
        for key in keys:
            if key.endswith(".scale_weight"):
                configs[key[: -len(".scale_weight")]] = {"format": "float8_e4m3fn", "legacy": True}
    return configs


def read_comfy_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """All tensors of a ComfyUI checkpoint plus its quantized-layer table.

    Marker tensors (``comfy_quant``, ``scaled_fp8``) are dropped; legacy ``scale_weight``
    is renamed to ``weight_scale``. fp8 weights stored as raw uint8 are viewed as fp8.
    """
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        layers = _layer_configs(handle, keys)
        for key in keys:
            if key.endswith((".comfy_quant", ".scaled_fp8")) or key == "scaled_fp8":
                continue
            if key.endswith((".scale_input", ".input_scale")):
                continue  # activation scales only matter to fp8 matmul kernels, not to dequant
            name = key[: -len(".scale_weight")] + ".weight_scale" if key.endswith(".scale_weight") else key
            tensors[name] = handle.get_tensor(key)

    for prefix, config in layers.items():
        fmt = config.get("format")
        if fmt not in FP8_FORMATS:
            raise ValueError(
                f"{path.name}: layer {prefix} uses quantization {fmt!r}; only fp8 scaled checkpoints "
                f"({', '.join(FP8_FORMATS)}) are supported here"
            )
        weight = tensors.get(f"{prefix}.weight")
        if weight is not None and weight.dtype is torch.uint8:
            tensors[f"{prefix}.weight"] = weight.view(FP8_FORMATS[fmt])
    return tensors, layers


CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def container_prefix(keys: Iterable[str], candidates: Iterable[str] = CONTAINER_PREFIXES) -> str:
    """The ComfyUI container prefix every key carries ("" if none), so a save can write it back."""
    keys = list(keys)
    for prefix in candidates:
        if keys and all(key.startswith(prefix) for key in keys):
            return prefix
    return ""


def strip_prefix(
    tensors: dict[str, torch.Tensor], layers: dict[str, dict[str, Any]], candidates: Iterable[str] = CONTAINER_PREFIXES
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Drop a ComfyUI container prefix if every key carries it."""
    prefix = container_prefix(tensors, candidates)
    if prefix:
        tensors = {key[len(prefix) :]: value for key, value in tensors.items()}
        layers = {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in layers.items()}
    return tensors, layers
