"""
ComfyUI "fp8_scaled" checkpoints: reading them, and training LoRAs on top of them.

ComfyUI ships many single-file checkpoints with their big Linear weights stored as
float8_e4m3fn plus one fp32 scale per tensor (dequantized weight = fp8 * scale). Three marker
styles exist in the wild, all handled here:

- ``_quantization_metadata`` JSON in the safetensors header: {"layers": {prefix: {"format": ...}}}
- ``<prefix>.comfy_quant`` uint8 tensors holding the same per-layer JSON
- legacy: a ``scaled_fp8`` marker tensor plus ``<prefix>.scale_weight`` (and ``scale_input``)

For training, the base keeps its fp8 storage -- a 12B DiT stays ~12 GB instead of ~24 GB in
bf16 -- and ``Fp8ScaledLinear`` dequantizes per call. The LoRA therefore learns against
exactly the weights ComfyUI runs at inference, rather than against a requantized copy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maltrainer.models.comfy_fp8_checkpoint import (
    CONTAINER_PREFIXES,
    FP8_FORMATS,
    container_prefix,
    read_comfy_checkpoint,
    strip_prefix,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONTAINER_PREFIXES",
    "FP8_FORMATS",
    "FP8_MAX",
    "Fp8ScaledLinear",
    "container_prefix",
    "dequantize",
    "dequantize_checkpoint",
    "quantize_rows",
    "read_comfy_checkpoint",
    "strip_prefix",
    "swap_in_fp8_linears",
]

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 weight -> (fp8 e4m3, per-output-row fp32 scale [out, 1])."""
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale


def dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp8 * per-tensor (or per-row) scale, rounded once into ``dtype``."""
    return (weight.float() * scale.float()).to(dtype)


def dequantize_checkpoint(tensors: dict[str, Tensor], layers: dict[str, Any], dtype: torch.dtype) -> dict[str, Tensor]:
    """Plain state dict with every fp8 layer dequantized (for small frozen encoders)."""
    out = dict(tensors)
    for prefix in layers:
        weight_key, scale_key = f"{prefix}.weight", f"{prefix}.weight_scale"
        if weight_key in out and scale_key in out:
            out[weight_key] = dequantize(out[weight_key], out.pop(scale_key), dtype)
    return {key: (value.to(dtype) if value.is_floating_point() else value) for key, value in out.items()}


class _Fp8LinearFunction(torch.autograd.Function):
    """y = x @ (fp8 * scale)^T + b, without ever saving the dequantized weight.

    The base is frozen, so only the input gradient is needed; it re-dequantizes in backward
    instead of keeping a bf16 copy of every weight alive for the whole step.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None, dtype: torch.dtype) -> Tensor:
        w = dequantize(weight, scale, dtype)
        ctx.save_for_backward(weight, scale)
        ctx.compute_dtype = dtype
        ctx.input_dtype = x.dtype
        return F.linear(x.to(dtype), w, None if bias is None else bias.to(dtype))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        weight, scale = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            w = dequantize(weight, scale, ctx.compute_dtype)
            grad_input = grad_output.to(ctx.compute_dtype).matmul(w).to(ctx.input_dtype)
        return grad_input, None, None, None, None


class Fp8ScaledLinear(nn.Module):
    """Frozen Linear backed by fp8 weights + a scale. Wrappable by LoRA like nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool, compute_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(Tensor, _Fp8LinearFunction.apply(x, self.weight, self.weight_scale, self.bias, self.compute_dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8 scaled"


def swap_in_fp8_linears(
    model: nn.Module,
    layers: dict[str, dict[str, Any]],
    compute_dtype: torch.dtype,
    key_map: Callable[[str], str] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
) -> int:
    """Replace every nn.Linear named in ``layers`` with an Fp8ScaledLinear of the same shape.

    Pass ``tensors`` to size each scale buffer from the checkpoint (per-tensor scalar or
    per-row [out, 1]); a strict load checks shapes even with ``assign=True``.
    """
    swapped = 0
    for prefix in layers:
        path = key_map(prefix) if key_map else prefix
        try:
            module = model.get_submodule(path)
        except AttributeError as e:
            raise ValueError(f"Quantized layer {prefix} has no module {path} in this model") from e
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Quantized layer {prefix} maps to {type(module).__name__}, expected Linear")
        replacement = Fp8ScaledLinear(module.in_features, module.out_features, module.bias is not None, compute_dtype)
        scale = (tensors or {}).get(f"{prefix}.weight_scale")
        if scale is not None and scale.shape != replacement.weight_scale.shape:
            replacement.weight_scale = torch.ones(scale.shape, dtype=torch.float32)
        parent_path, _, attribute = path.rpartition(".")
        setattr(model.get_submodule(parent_path) if parent_path else model, attribute, replacement)
        swapped += 1
    return swapped
