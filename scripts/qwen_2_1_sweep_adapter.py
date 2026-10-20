"""Checkpoint naming and adapter swapping for scripts/qwen_2_1_sweep_checkpoints.py."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork


def sort_key(path: Path) -> tuple[int, int]:
    """`..._step_600` -> 600; the unsuffixed final checkpoint sorts last."""
    match = re.search(r"_step_(\d+)$", path.stem)
    return (0, int(match.group(1))) if match else (1, 0)


def label_of(path: Path) -> str:
    match = re.search(r"_step_(\d+)$", path.stem)
    return f"step {match.group(1)}" if match else "final"


def swap_adapter(network: QwenImage21LoRANetwork, path: Path, scale: float) -> None:
    """Copy one checkpoint's weights into the adapter layers already attached to `network`.

    Every snapshot of a run shares rank and module list, so the attached layers never need
    rebuilding. `scale` multiplies alpha/rank; 0 reproduces the base model exactly.
    """
    from safetensors.torch import load_file

    layers = network.assistant_layers
    state = load_file(str(path))
    for key in state:
        if not key.endswith(".lora_A.weight"):
            continue
        module_path = key[len(network.prefix) : -len(".lora_A.weight")]
        layer: Any = layers[module_path.replace(".", "_")]
        layer.lora_down.weight.data.copy_(state[key].to(layer.lora_down.weight.dtype))
        up = state[key.replace(".lora_A.", ".lora_B.")]
        layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
        alpha = state.get(key.replace(".lora_A.weight", ".alpha"))
        rank = state[key].shape[0]
        layer.scale = (float(alpha.item()) if alpha is not None else float(rank)) / rank * scale
