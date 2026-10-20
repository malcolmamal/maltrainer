"""Reading a frozen assistant adapter file and turning one of its modules into a LoRA layer."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.qwen_image_2_1.lora_layer import QwenImage21LoRALayer

_SLOT_SUFFIXES = (
    (".lora_A.weight", "down"),
    (".lora_down.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_up.weight", "up"),
    (".alpha", "alpha"),
)


def read_adapter_pairs(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """The adapter's tensors grouped per module name into ``down`` / ``up`` / ``alpha`` slots."""
    state = load_file(str(path))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        for suffix, slot in _SLOT_SUFFIXES:
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[slot] = value
                break
    return pairs


def find_linear(transformer: nn.Module, module_path: str) -> nn.Linear | None:
    """The ``nn.Linear`` at ``module_path``, or None when it is missing or not a linear."""
    try:
        module = transformer.get_submodule(module_path)
    except AttributeError:
        return None
    return module if isinstance(module, nn.Linear) else None


def build_assistant_layer(
    module: nn.Linear,
    parts: dict[str, torch.Tensor],
    multiplier: float,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> tuple[QwenImage21LoRALayer, int]:
    """A frozen LoRA layer holding one adapter module's weights, and its rank."""
    down, up = parts["down"], parts["up"]
    rank = down.shape[0]
    alpha = float(parts["alpha"].item()) if "alpha" in parts else float(rank)
    layer = QwenImage21LoRALayer(module, rank, alpha)
    layer.lora_down.weight.data.copy_(down.to(layer.lora_down.weight.dtype))
    layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
    layer.scale = (alpha / rank) * multiplier
    layer.requires_grad_(False)
    layer.eval()
    if dtype is not None or device is not None:
        layer.to(device=device, dtype=dtype)
    return layer, rank
