"""
INT8 ConvRot support for MiniMax H3 transformer checkpoints.
Provides INT8 quantized matrix multiplication with Hadamard activation rotation.

The rotation maths lives in ``int8_rotation`` and the autograd functions in ``int8_functions``; every name is
re-exported here (the tests patch the function classes on this module, where ``_int8_linear_forward`` looks them up).
"""

from __future__ import annotations

import json
import logging
from types import MethodType
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn

from maltrainer.models.minimax_h3.int8_functions import (
    _GRAD_COMPUTE_DTYPE,
    DEFAULT_GRAD_PRECISION,
    GRAD_PRECISIONS,
    _Int8ConvRotFunction,
    _resolve_grad_precision,
    _TransientInt8ConvRotFunction,
)
from maltrainer.models.minimax_h3.int8_rotation import (
    _HADAMARD_CACHE,
    _hadamard,
    _int_mm,
    _is_power_of_four,
    _quantize_rows,
    rotate_activation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRAD_PRECISION",
    "GRAD_PRECISIONS",
    "_GRAD_COMPUTE_DTYPE",
    "_HADAMARD_CACHE",
    "_Int8ConvRotFunction",
    "_TransientInt8ConvRotFunction",
    "_hadamard",
    "_int8_linear_forward",
    "_int_mm",
    "_is_power_of_four",
    "_quantize_rows",
    "_resolve_grad_precision",
    "enable_int8_convrot",
    "load_comfy_int8_convrot_state_dict",
    "parse_comfy_quant_marker",
    "prepare_int8_convrot_modules",
    "rotate_activation",
]

_T = torch.Tensor


def parse_comfy_quant_marker(marker: torch.Tensor) -> dict[str, Any]:
    try:
        payload = bytes(marker.detach().cpu().flatten().tolist()).decode("utf-8")
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Invalid Comfy quantization marker") from error
    if not isinstance(value, dict):
        raise ValueError("Comfy quantization marker must contain a JSON object")
    return value


def _int8_linear_forward(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    group_size = int(cast(torch.Tensor, module.int8_convrot_groupsize).item())
    operation = _Int8ConvRotFunction if inputs.is_cuda and hasattr(torch, "_int_mm") else _TransientInt8ConvRotFunction
    precision = getattr(module, "int8_convrot_grad_precision", DEFAULT_GRAD_PRECISION)
    y: torch.Tensor = operation.apply(inputs, module.weight, module.scale_weight, module.bias, group_size, precision)
    return y


def load_comfy_int8_convrot_state_dict(checkpoint_path: str, *, device: torch.device) -> tuple[dict[str, _T], int]:
    state_dict: dict[str, torch.Tensor] = {}
    marker_config: dict[str, dict[str, Any]] = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            if key.endswith(".comfy_quant"):
                base = key[: -len(".comfy_quant")]
                config = parse_comfy_quant_marker(handle.get_tensor(key))
                if config.get("format") != "int8_tensorwise" or config.get("convrot") is not True:
                    raise ValueError(f"{key} is not an INT8 ConvRot marker")
                group_size = int(config.get("convrot_groupsize", 0))
                if not _is_power_of_four(group_size):
                    raise ValueError(f"{key} contains invalid ConvRot group size {group_size}")
                marker_config[base] = config

        scale_bases = {key[: -len(".weight_scale")] for key in keys if key.endswith(".weight_scale")}
        weight_bases = {
            key[: -len(".weight")]
            for key in keys
            if key.endswith(".weight") and handle.get_slice(key).get_dtype() == "I8"
        }
        if not marker_config or set(marker_config) != scale_bases or set(marker_config) != weight_bases:
            raise ValueError(
                "INT8 ConvRot checkpoint has inconsistent marker, scale, and quantized-weight sets: "
                f"markers={len(marker_config)}, scales={len(scale_bases)}, weights={len(weight_bases)}"
            )

        for key in keys:
            if key.endswith(".comfy_quant"):
                continue
            value = handle.get_tensor(key)
            if key.endswith(".weight_scale"):
                base = key[: -len(".weight_scale")]
                scale = value.float().reshape(-1, 1)
                state_dict[f"{base}.scale_weight"] = scale.to(device)
                state_dict[f"{base}.int8_convrot_groupsize"] = torch.tensor(
                    int(marker_config[base]["convrot_groupsize"]), device=device, dtype=torch.int32
                )
            else:
                state_dict[key] = value.to(device)
    return state_dict, len(marker_config)


def prepare_int8_convrot_modules(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    scale_shapes = {
        key[: -len(".scale_weight")]: tuple(value.shape)
        for key, value in state_dict.items()
        if key.endswith(".scale_weight")
    }
    registered = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in scale_shapes:
            continue
        group_key = f"{name}.int8_convrot_groupsize"
        if group_key not in state_dict:
            raise ValueError(f"INT8 ConvRot state is missing {group_key}")
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.register_buffer("scale_weight", torch.ones(scale_shapes[name], dtype=torch.float32))
        module.register_buffer("int8_convrot_groupsize", torch.zeros((), dtype=torch.int32))
        registered += 1
    if registered != len(scale_shapes):
        raise ValueError(
            f"INT8 ConvRot checkpoint targets {len(scale_shapes)} linears, but the model exposes {registered}"
        )
    return registered


def enable_int8_convrot(model: nn.Module, grad_precision: str | None = None) -> int:
    precision = _resolve_grad_precision(grad_precision)
    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or not hasattr(module, "scale_weight"):
            continue
        if module.weight.dtype is not torch.int8:
            raise TypeError("INT8 ConvRot scale is attached to a non-INT8 Linear weight")
        group_size = int(cast(torch.Tensor, module.int8_convrot_groupsize).item())
        if module.weight.shape[1] % group_size:
            raise ValueError(
                f"INT8 ConvRot group size {group_size} does not divide {module.weight.shape[1]} input features"
            )
        cast(Any, module).int8_convrot_grad_precision = precision
        cast(Any, module).forward = MethodType(_int8_linear_forward, module)
        patched += 1
    logger.info("Enabled H3 INT8 ConvRot runtime on %d Linear layers (backward precision: %s)", patched, precision)
    if precision == "int8":
        logger.warning(
            "INT8 backward re-quantizes the gradient in every one of the %d layers; expect "
            "the LoRA to partly fit quantization noise. Use bf16 unless benchmarking.",
            patched,
        )
    return patched
