"""What the H3 Image finetune trains: the windows of block Linears, and the small always-trained modules.

H3 Image has 50 blocks of ~385M parameters (attention qkv 116M and out 39M, MLP fc1 154M and fc2 77M), 19.3B in all,
stored as INT8 ConvRot. A window is a run of ``finetune.blocks_per_window`` blocks (default 5, 1.9B), after Fizgig's H3
finetune, which rotates over blocks. ``finetune.block_range: [20, 49]`` limits the run to the blocks where identity is
made, as the LoRA's ``block_range`` does. The token refiner, the input and output layers, the norms and the AdaLN
stay frozen unless ``finetune.always_on`` lists their modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import torch
from torch import nn

from maltrainer.finetune.rotation_int8 import Window

DEFAULT_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
DEFAULT_BLOCKS_PER_WINDOW = 5


def _block_and_leaf(key: str) -> tuple[int, str] | None:
    """``blocks.12.attn.qkv_proj.weight`` -> (12, ``attn.qkv_proj``); None for any other key."""
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit() or parts[-1] != "weight":
        return None
    return int(parts[1]), ".".join(parts[2:-1])


def plan_windows(keys: Iterable[str], finetune: dict[str, Any]) -> list[Window]:
    """The windows for the INT8 Linears ``keys`` (see the module doc); ``finetune`` is the config's section."""
    per_window = int(finetune.get("blocks_per_window") or DEFAULT_BLOCKS_PER_WINDOW)
    if per_window < 1:
        raise ValueError(f"finetune.blocks_per_window must be at least 1, got {per_window}")
    targets = tuple(finetune.get("targets") or DEFAULT_TARGETS)
    block_range = finetune.get("block_range")
    by_block: dict[int, list[str]] = {}
    for key in keys:
        found = _block_and_leaf(key)
        if found is None or found[1] not in targets:
            continue
        if block_range is not None and not int(block_range[0]) <= found[0] <= int(block_range[1]):
            continue
        by_block.setdefault(found[0], []).append(key)
    if not by_block:
        raise ValueError(f"finetune.targets {list(targets)} and block_range {block_range} match no INT8 Linear")
    blocks = sorted(by_block)
    windows: list[Window] = []
    for start in range(0, len(blocks), per_window):
        chunk = blocks[start : start + per_window]
        windows.append((f"blocks {chunk[0]}-{chunk[-1]}", [key for block in chunk for key in by_block[block]]))
    return windows


def always_on_params(model: nn.Module, modules: Sequence[str]) -> list[tuple[str, nn.Parameter]]:
    """(name, weight) of every Linear under ``modules``, made trainable (weights only, in bf16 or fp32)."""
    out: list[tuple[str, nn.Parameter]] = []
    for module_name in modules:
        for name, module in model.get_submodule(module_name).named_modules():
            path = f"{module_name}.{name}" if name else module_name
            if not isinstance(module, nn.Linear):
                continue
            if hasattr(module, "scale_weight"):
                raise ValueError(f"{path} is an INT8 block Linear: the windows train it, not finetune.always_on")
            if module.weight.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"{path} is stored in {module.weight.dtype}; always-on weights must be bf16 or fp32")
            module.weight.requires_grad_(True)
            out.append((f"{path}.weight", cast(nn.Parameter, module.weight)))
    if modules and not out:
        raise ValueError(f"finetune.always_on {list(modules)} matched no Linear")
    return out
