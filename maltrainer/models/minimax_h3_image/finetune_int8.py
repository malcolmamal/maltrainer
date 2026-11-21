"""An INT8 ConvRot Linear that can also hold a trained dense weight, for the H3 Image finetune's rotating windows.

The same ``nn.Linear`` object is changed in place (its weight and scale), never swapped for another module: a frozen
training adapter wraps the Linear's ``forward`` (``MiniMaxH3LoRANetwork._wrap_forward``), and that wrapper would be
lost with a new module. ``linear_forward`` picks the INT8 kernels or a plain matmul by the weight's dtype.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.int8_convrot import _int8_linear_forward
from maltrainer.models.minimax_h3.int8_rotation import _quantize_rows, dequantize_weight, rotate_activation


def linear_forward(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    """The INT8 ConvRot kernels while the weight is int8 codes, an ordinary matmul while it is a trained dense one."""
    if module.weight.dtype is torch.int8:
        return _int8_linear_forward(module, inputs)
    return F.linear(inputs, module.weight, module.bias)


def make_swappable(model: nn.Module) -> list[str]:
    """Route every INT8 ConvRot Linear through ``linear_forward``; their weight keys (``blocks.N.mlp.fc1.weight``)."""
    keys = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "scale_weight"):
            cast(Any, module).forward = MethodType(linear_forward, module)
            keys.append(f"{name}.weight")
    return keys


def group_size(module: nn.Linear) -> int:
    return int(cast(torch.Tensor, module.int8_convrot_groupsize).item())


def decode(module: nn.Linear, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The dense weight an INT8 Linear stands for (on the module's device)."""
    scale = cast(torch.Tensor, module.scale_weight)
    return dequantize_weight(module.weight.detach(), scale, group_size(module), dtype)


def encode(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 codes and the per-row scale of a dense weight, in the checkpoint's rotated layout."""
    return _quantize_rows(rotate_activation(weight.float(), group))


def unfreeze(module: nn.Linear, weight: torch.Tensor) -> nn.Parameter:
    """Make ``weight`` (dense, on the module's device) the module's trainable weight."""
    module.weight = parameter = nn.Parameter(weight, requires_grad=True)
    return parameter


def freeze(module: nn.Linear, weight: torch.Tensor) -> None:
    """Quantize ``weight`` back into the module as frozen INT8 codes with their scale."""
    codes, scale = encode(weight, group_size(module))
    module.weight = nn.Parameter(codes, requires_grad=False)
    cast(Any, module).scale_weight = scale.to(cast(torch.Tensor, module.scale_weight).dtype)
