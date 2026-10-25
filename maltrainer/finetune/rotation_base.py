"""The original model in place under a window rotator (rotation.py), for a while: prior preservation records the
predictions of images added on a resume from it (maltrainer/trainers/finetune/prior_setup.py).

Every window must be frozen. Each rotating weight is then an fp8 copy quantized from its master; for the duration it
is quantized from the base file's tensor instead, and afterwards from the master again. The fp8 copy was made that
way from that master (on load from the resume state's bf16 file, and on every rotation), so it comes back bit for
bit. The always-trained weights (the text fusion) are copied aside on the CPU and put back exactly. A base file in
fp8 (ComfyUI's krea2_raw_fp8_scaled, the text fusion included) is dequantized tensor by tensor. Only one rotating
weight's base tensor is in memory at a time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn

from maltrainer.finetune.prior import CONTAINER_PREFIXES
from maltrainer.finetune.rotation import WindowRotator
from maltrainer.models.comfy_fp8 import dequantize, quantize_rows


def _freeze_to(rotator: WindowRotator, key: str, weight: torch.Tensor) -> None:
    """The frozen fp8 Linear of ``key`` re-quantized from ``weight`` (bf16, any device)."""
    module: Any = rotator.model.get_submodule(key[: -len(".weight")])
    module.weight, module.weight_scale = quantize_rows(weight.to(rotator.device, torch.bfloat16))


def _base_tensor(handle: Any, keys: set[str], key: str, prefixes: Sequence[str], path: Path) -> torch.Tensor:
    """``key``'s weight in the base file (bf16, or fp8 with its scale), as bf16 on the CPU."""
    stored = next((prefix + key for prefix in prefixes if prefix + key in keys), None)
    if stored is None:
        raise KeyError(f"{path.name} has no tensor for {key}")
    scale = stored[: -len(".weight")] + ".weight_scale"
    if scale in keys:
        return dequantize(handle.get_tensor(stored), handle.get_tensor(scale), torch.bfloat16)
    return cast(torch.Tensor, handle.get_tensor(stored).to(torch.bfloat16))


@contextlib.contextmanager
def original_weights(
    rotator: WindowRotator,
    always_on: Sequence[tuple[str, nn.Parameter]],
    path: str | Path,
    prefixes: Sequence[str] = CONTAINER_PREFIXES,
) -> Iterator[None]:
    """The rotating weights and ``always_on`` hold the base file's values for the duration, then their trained
    values again, bit for bit. Every window must be frozen (``rotator.current`` None)."""
    if rotator.current is not None:
        raise RuntimeError("the original weights go in with every window frozen")
    path = Path(path)
    trained = [param.detach().to("cpu", copy=True) for _, param in always_on]  # the text fusion: small
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle, torch.no_grad():
            keys = set(handle.keys())
            for key in rotator.master:
                _freeze_to(rotator, key, _base_tensor(handle, keys, key, prefixes, path))
            for name, param in always_on:
                param.copy_(_base_tensor(handle, keys, name, prefixes, path).to(param.device, param.dtype))
        yield
    finally:
        with torch.no_grad():
            for key in rotator.master:
                _freeze_to(rotator, key, rotator.master[key])
            for (_, param), saved in zip(always_on, trained, strict=True):
                param.copy_(saved.to(param.device))
