"""The INT8 ConvRot linear as autograd functions (INT8 kernel path and eager fallback)."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from maltrainer.models.minimax_h3.int8_rotation import _int_mm, _quantize_rows, rotate_activation

_T = torch.Tensor


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
