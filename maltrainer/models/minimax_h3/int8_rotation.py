"""Hadamard activation rotation and row-wise INT8 quantisation for INT8 ConvRot."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


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
