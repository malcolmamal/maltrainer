"""
torch.compile for a DiT's repeated blocks.

Each block is compiled on its own (``nn.Module.compile``), so gradient checkpointing stays outside
the compiled region: ``checkpoint(block, ...)`` calls the compiled forward, and the backward
recomputes through it. Compile after any LoRA wrapping, since the wrappers replace Linear forwards
that the compiled graph then traces.

Inductor needs Triton. Without it (e.g. a Windows venv with no ``triton-windows``) this logs why
and leaves the blocks eager, so a config with compile on still trains.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Iterable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def compile_unavailable_reason(device: torch.device) -> Optional[str]:
    """Why inductor cannot compile for ``device``, or None."""
    if not hasattr(torch, "compile"):
        return f"torch {torch.__version__} has no torch.compile"
    if device.type == "cuda" and importlib.util.find_spec("triton") is None:
        return "Triton is not installed (on Windows: pip install triton-windows)"
    return None


def compile_blocks(blocks: Iterable[nn.Module], device: torch.device, mode: Optional[str] = None) -> bool:
    """Compile every block in place; returns whether it did."""
    reason = compile_unavailable_reason(device)
    blocks = list(blocks)
    if reason:
        logger.warning("train.compile is on but the blocks stay eager: %s", reason)
        return False
    # Caption lengths and buckets vary; the first new shape recompiles with dynamic sizes, then
    # later shapes reuse that graph. The limit leaves room for a few static specialisations.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    for block in blocks:
        block.compile(mode=mode)
    logger.info("torch.compile: %d blocks (mode %s); the first steps of each new shape compile", len(blocks), mode or "default")
    return True
