"""Reading a Qwen-Image 2.1 transformer file and moving the built model to its device and dtype."""

from __future__ import annotations

import torch
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import load_comfy_int8_convrot_state_dict


def read_state_dict(path: str) -> tuple[dict[str, torch.Tensor], int]:
    """The checkpoint's tensors on the CPU and how many linears are INT8 ConvRot (0 for a plain file)."""
    if is_int8_convrot(path):
        return load_comfy_int8_convrot_state_dict(path, device=torch.device("cpu"))
    return load_file(path, device="cpu"), 0


def to_keeping_scales(model: nn.Module, device: str | torch.device, dtype: torch.dtype) -> None:
    """``model.to(device, dtype)`` except that the float32 quantization scales are not rounded to ``dtype``."""
    scales = {name: module.scale_weight for name, module in model.named_modules() if hasattr(module, "scale_weight")}
    model.to(device=device, dtype=dtype)
    for name, scale in scales.items():
        model.get_submodule(name).scale_weight = scale.to(device)
