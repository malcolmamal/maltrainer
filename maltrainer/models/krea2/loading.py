"""
Load a Krea 2 DiT from a ComfyUI single file (fp8_scaled or bf16) for LoRA training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from accelerate import init_empty_weights

from maltrainer.models.comfy_fp8 import (
    container_prefix,
    dequantize,
    quantize_rows,
    read_comfy_checkpoint,
    strip_prefix,
    swap_in_fp8_linears,
)
from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer

logger = logging.getLogger(__name__)


def is_block_linear(key: str) -> bool:
    return (
        key.startswith("blocks.")
        and key.endswith(".weight")
        and (".attn.w" in key or ".attn.gate" in key or ".mlp." in key)
    )


def load_krea2_transformer(
    path: str | Path,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    master: dict[str, torch.Tensor] | None = None,
) -> tuple[Krea2Transformer, dict[str, object]]:
    """Build the DiT with fp8 block Linears.

    An fp8_scaled file keeps the exact fp8 weights ComfyUI runs. A bf16 file (e.g. Comfy-Org's
    krea2_raw_bf16) has its 224 block Linears quantized per output row to fp8 on load -- the
    same set Fizgig quantizes -- which keeps the 12.9B model at ~13 GB instead of ~26 GB; the
    small text-fusion and I/O layers stay bf16.

    ``master``: a dict to fill with every block Linear's weight in bf16 on the CPU, for a full
    finetune (maltrainer/finetune/rotation.py). From a bf16 file these are the file's own tensors
    (no copy); from an fp8 file they are dequantized, so they carry fp8's rounding.
    """
    tensors, layers = read_comfy_checkpoint(path)
    prefix_found = container_prefix(tensors)
    tensors, layers = strip_prefix(tensors, layers)
    info: dict[str, object] = {"source_fp8_layers": len(layers), "key_prefix": prefix_found}

    if master is not None:
        for key in [k for k in tensors if is_block_linear(k)]:
            prefix = key[: -len(".weight")]
            if prefix in layers:
                master[key] = dequantize(tensors[key], tensors[f"{prefix}.weight_scale"], torch.bfloat16)
            else:
                master[key] = tensors[key].to(torch.bfloat16)

    if not layers:
        for key in [k for k in tensors if is_block_linear(k)]:
            prefix = key[: -len(".weight")]
            fp8, scale = quantize_rows(tensors[key].to(device))  # on the GPU: ~12B elements
            tensors[key], tensors[f"{prefix}.weight_scale"] = fp8, scale
            layers[prefix] = {"format": "float8_e4m3fn", "quantized_on_load": True}
        info["quantized_on_load"] = len(layers)

    for key, value in list(tensors.items()):
        if (
            value.is_floating_point()
            and value.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not key.endswith("weight_scale")
        ):
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
        Path(path).name,
        len(layers),
        "quantized on load" if info.get("quantized_on_load") else "as stored",
        compute_dtype,
    )
    return model, info
