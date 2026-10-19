"""Tiny LoRA files for the script tests."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

KOHYA = (".lora_down.weight", ".lora_up.weight")
PEFT = (".lora_A.weight", ".lora_B.weight")


def write_lora(path: Path, modules: dict, layout=PEFT, metadata: dict | None = None) -> Path:
    """modules: name -> (down [r, in], up [out, r], alpha or None for 'no alpha tensor')."""
    state = {}
    for name, (down, up, alpha) in modules.items():
        state[name + layout[0]] = down.float().contiguous()
        state[name + layout[1]] = up.float().contiguous()
        if alpha is not None:
            state[name + ".alpha"] = torch.tensor(float(alpha))
    save_file(state, str(path), metadata=metadata)
    return path


def delta(down: torch.Tensor, up: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
    """What a LoRA applies: alpha / rank * up @ down (alpha missing = rank)."""
    rank = down.shape[0]
    return (1.0 if alpha is None else alpha / rank) * up.double() @ down.double()
