"""
INT8 ConvRot support for MiniMax H3 transformer checkpoints.
Provides INT8 quantized matrix multiplication with Hadamard activation rotation.
"""

from __future__ import annotations

import json
import logging
import math
from types import MethodType
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

_T = torch.Tensor
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def parse_comfy_quant_marker(marker: torch.Tensor) -> dict[str, Any]:
    try:
        payload = bytes(marker.detach().cpu().flatten().tolist()).decode("utf-8")
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Invalid Comfy quantization marker") from error
    if not isinstance(value, dict):
        raise ValueError("Comfy quantization marker must contain a JSON object")
    return value


def _is_power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value > 1 and value % 4 == 0:
        value //= 4
    return value == 1


def _hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _is_power_of_four(size):
        raise ValueError(f"INT8 ConvRot group size must be a power of four, got {size}")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    base = torch.tensor(((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)), device=device, dtype=dtype)
    matrix = base
    current = 4
    while current < size:
        matrix = torch.kron(matrix, base)
        current *= 4
    matrix = matrix / math.sqrt(size)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def rotate_activation(value: torch.Tensor, group_size: int) -> torch.Tensor:
    if value.shape[-1] % group_size:
        raise ValueError(f"INT8 ConvRot group size {group_size} does not divide {value.shape[-1]} input features")
    matrix = _hadamard(group_size, value.device, value.dtype)
    grouped = value.reshape(-1, value.shape[-1] // group_size, group_size)
    return torch.matmul(grouped, matrix).reshape(value.shape)


def _quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (value.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    quantized = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _int_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("INT8 ConvRot requires torch._int_mm on CUDA")
    rows = left.shape[0]
    if rows > 16:
        return torch._int_mm(left.contiguous(), right)
    padded = F.pad(left, (0, 0, 0, 17 - rows))
    return torch._int_mm(padded.contiguous(), right)[:rows]


# How the input gradient is computed through a frozen INT8 ConvRot layer. The forward pass
# is always INT8 (it has to reproduce Comfy's kernel numerics); only the backward varies.
#
#   fp32  most accurate, slowest. ~1.2 s/step on a 5090 at 512px rank 64.
#   bf16  matches musubi-tuner's reference trainer, roughly halves the backward cost.
#   int8  fastest, uses INT8 tensor cores, but re-quantizes the gradient once per layer.
#         H3 stacks 50 blocks plus a token refiner, so that error compounds and the LoRA
#         partly fits quantization noise. Kept for A/B only; not recommended.
GRAD_PRECISIONS = ("bf16", "fp32", "int8")
DEFAULT_GRAD_PRECISION = "bf16"

_GRAD_COMPUTE_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _resolve_grad_precision(value: str | None) -> str:
    precision = str(value or DEFAULT_GRAD_PRECISION).lower()
    aliases = {"float32": "fp32", "fp32": "fp32", "bfloat16": "bf16", "bf16": "bf16", "int8": "int8"}
    if precision not in aliases:
        raise ValueError(f"Unknown INT8 ConvRot gradient precision {value!r}; expected one of {GRAD_PRECISIONS}")
    return aliases[precision]


class _Int8ConvRotFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: _T, weight: _T, scale: _T, bias: _T | None, group_size: int, grad_precision: str
    ) -> _T:
        group_size = int(group_size)
        rotated = rotate_activation(inputs, group_size)
        flat = rotated.reshape(-1, rotated.shape[-1])
        quantized, input_scale = _quantize_rows(flat)
        output = _int_mm(quantized, weight.t())
        output = output.float() * input_scale * scale.reshape(1, -1)
        if bias is not None:
            output = output + bias.float()
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = inputs.dtype
        ctx.input_shape = inputs.shape
        ctx.group_size = group_size
        ctx.grad_precision = grad_precision
        return output.to(inputs.dtype).reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx: Any, grad_output: _T) -> tuple[_T | None, ...]:
        weight, scale = ctx.saved_tensors
        precision = ctx.grad_precision

        if precision == "int8":
            folded = grad_output.reshape(-1, grad_output.shape[-1]).float() * scale.reshape(1, -1)
            quantized, grad_scale = _quantize_rows(folded)
            grad_input = _int_mm(quantized, weight).float() * grad_scale
        else:
            compute_dtype = _GRAD_COMPUTE_DTYPE[precision]
            folded = grad_output.reshape(-1, grad_output.shape[-1]).to(compute_dtype) * scale.reshape(1, -1).to(
                compute_dtype
            )
            grad_input = torch.zeros((folded.shape[0], weight.shape[1]), dtype=compute_dtype, device=folded.device)
            # Dequantize bounded row chunks so the frozen weight never expands to float in full.
            chunk = max(1, 2**22 // max(weight.shape[1], 1))
            for start in range(0, weight.shape[0], chunk):
                stop = min(start + chunk, weight.shape[0])
                grad_input.addmm_(folded[:, start:stop], weight[start:stop].to(compute_dtype))

        grad_input = rotate_activation(grad_input.to(ctx.input_dtype), ctx.group_size)
        return grad_input.reshape(ctx.input_shape), None, None, None, None, None


class _TransientInt8ConvRotFunction(torch.autograd.Function):
    """Eager fallback for CPU or builds without torch._int_mm. Always computes in float."""

    @staticmethod
    def forward(
        ctx: Any, inputs: _T, weight: _T, scale: _T, bias: _T | None, group_size: int, _grad_precision: object = None
    ) -> _T:
        group_size = int(group_size)
        rotated = rotate_activation(inputs, group_size)
        dense_weight = (weight.float() * scale.float()).to(inputs.dtype)
        output = F.linear(rotated, dense_weight, bias.to(inputs.dtype) if bias is not None else None)
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = inputs.dtype
        ctx.input_shape = inputs.shape
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: _T) -> tuple[_T | None, ...]:
        weight, scale = ctx.saved_tensors
        dense_weight = weight.float() * scale.float()
        grad_input = grad_output.reshape(-1, grad_output.shape[-1]).float() @ dense_weight
        grad_input = rotate_activation(grad_input.to(ctx.input_dtype), ctx.group_size)
        return grad_input.reshape(ctx.input_shape), None, None, None, None, None


def _int8_linear_forward(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    group_size = int(cast(torch.Tensor, module.int8_convrot_groupsize).item())
    operation = _Int8ConvRotFunction if inputs.is_cuda and hasattr(torch, "_int_mm") else _TransientInt8ConvRotFunction
    grad_precision = getattr(module, "int8_convrot_grad_precision", DEFAULT_GRAD_PRECISION)
    return operation.apply(inputs, module.weight, module.scale_weight, module.bias, group_size, grad_precision)


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
