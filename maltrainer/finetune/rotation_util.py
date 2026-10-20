"""Pure helpers of the window rotator (``rotation``): block-leaf names, CPU copies, optimizer hooks."""

from __future__ import annotations

from typing import Any

import torch


def _block_leaf(key: str, blocks: str) -> str | None:
    """``blocks.12.attn.wq.weight`` -> ``attn.wq`` (None outside the numbered blocks)."""
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != blocks or not parts[1].isdigit() or parts[-1] != "weight":
        return None
    return ".".join(parts[2:-1])


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value


def _remove_hooks(optimizer: Any) -> None:
    """Fused optimizers (Automagic3) step from grad hooks; a frozen window's must not fire."""
    for handle in getattr(optimizer, "_hook_handles", []):
        handle.remove()
    if hasattr(optimizer, "_hook_handles"):
        optimizer._hook_handles = []
