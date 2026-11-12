"""The noise levels MiniMax H3 Image was distilled on, and where a checkpoint states them."""

from __future__ import annotations

import json
import struct
from pathlib import Path

# MiniMaxAI/MiniMax-H3-Image pdd_config.json, and the Comfy-Org file's metadata (config.transformer.pdd_sigmas).
PDD_SIGMAS: tuple[float, ...] = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)


def file_pdd_sigmas(path: str | Path) -> tuple[float, ...]:
    """The sigma grid in a safetensors file's metadata (``PDD_SIGMAS`` when the file does not say)."""
    try:
        with Path(path).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            metadata = json.loads(handle.read(size)).get("__metadata__") or {}
        sigmas = (json.loads(metadata.get("config") or "{}").get("transformer") or {}).get("pdd_sigmas")
    except (OSError, ValueError, struct.error):
        return PDD_SIGMAS
    if not sigmas or len(sigmas) < 2:
        return PDD_SIGMAS
    return tuple(float(value) for value in sigmas)
