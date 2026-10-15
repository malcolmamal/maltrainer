"""
Load a Krea 2 DiT from a ComfyUI single file (fp8_scaled or bf16) for LoRA training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from accelerate import init_empty_weights

from maltrainer.models.comfy_fp8 import read_comfy_checkpoint, strip_prefix, swap_in_fp8_linears
from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer

logger = logging.getLogger(__name__)

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448


def _quantize_rows(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 weight -> (fp8 e4m3, per-output-row fp32 scale)."""
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale


def _is_block_linear(key: str) -> bool:
    return key.startswith("blocks.") and key.endswith(".weight") and (".attn.w" in key or ".attn.gate" in key or ".mlp." in key)


def load_krea2_transformer(path: Union[str, Path], device: torch.device, compute_dtype: torch.dtype = torch.bfloat16) -> Tuple[Krea2Transformer, Dict[str, object]]:
    """Build the DiT with fp8 block Linears.

    An fp8_scaled file keeps the exact fp8 weights ComfyUI runs. A bf16 file (e.g. Comfy-Org's
    krea2_raw_bf16) has its 224 block Linears quantized per output row to fp8 on load -- the
    same set Fizgig quantizes -- which keeps the 12.9B model at ~13 GB instead of ~26 GB; the
    small text-fusion and I/O layers stay bf16.
    """
    tensors, layers = read_comfy_checkpoint(path)
    tensors, layers = strip_prefix(tensors, layers)
    info: Dict[str, object] = {"source_fp8_layers": len(layers)}

    if not layers:
        for key in [k for k in tensors if _is_block_linear(k)]:
            prefix = key[: -len(".weight")]
            fp8, scale = _quantize_rows(tensors[key].to(device))  # on the GPU: ~12B elements
            tensors[key], tensors[f"{prefix}.weight_scale"] = fp8, scale
            layers[prefix] = {"format": "float8_e4m3fn", "quantized_on_load": True}
        info["quantized_on_load"] = len(layers)

    for key, value in list(tensors.items()):
        if value.is_floating_point() and value.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) and not key.endswith("weight_scale"):
            tensors[key] = value.to(compute_dtype)

    with init_empty_weights(include_buffers=True):
        model = Krea2Transformer(Krea2Config())
    swap_in_fp8_linears(model, layers, compute_dtype, tensors=tensors)
    result = model.load_state_dict(tensors, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} does not match the Krea 2 DiT "
            f"(missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]})"
        )
    model.compute_dtype = compute_dtype
    model.requires_grad_(False)
    model.to(device)
    info["fp8_linears"] = len(layers)
    logger.info(
        "Krea 2 DiT from %s: %d fp8 Linears (%s), rest %s",
        Path(path).name, len(layers), "quantized on load" if info.get("quantized_on_load") else "as stored",
        compute_dtype,
    )
    return model, info
