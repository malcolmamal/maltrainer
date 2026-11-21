"""
Loading ComfyUI's Fibo 2.0 single-file DiT: ``fibo2_dev_bf16.safetensors``, or Comfy-Org's INT8 ConvRot repack of
the same Dev model, ``fibo2_dev_int8_convrot.safetensors`` (8.4 GB).

An INT8 file loads two ways (``int8_load``): ``decode`` turns every int8 Linear back into a dense bf16 weight, so the
model trains exactly as from the bf16 file and needs the same VRAM (only the download is smaller); ``runtime`` keeps
the Linears int8 on the GPU and runs the int8 kernels shared with MOSS-Image and FLUX.3 (about half the VRAM).

The INT8 repack of Edit Turbo (``fibo2_edit_turbo_*``) is the step-distilled model, which BRIA says is not a
fine-tuning base: it is refused, and a bf16 Edit Turbo file only gets a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from maltrainer.models.fibo2.config import Fibo2Config
from maltrainer.models.fibo2.model import Fibo2Transformer
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    parse_comfy_quant_marker,
    prepare_int8_convrot_modules,
)
from maltrainer.models.minimax_h3.int8_rotation import dequantize_weight

logger = logging.getLogger(__name__)

CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def is_quantized(path: str | Path) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(key.endswith(".comfy_quant") for key in handle.keys())


def strip_container_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in CONTAINER_PREFIXES:
        if state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return state


INT8_LOADS = ("decode", "runtime")
SKIPPED = (".comfy_quant", ".weight_scale", ".comfy_attention.config")


def _decoded_state(path: Path, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], int]:
    """(every tensor of an INT8 ConvRot file, its int8 Linears decoded to dense ``dtype`` weights; how many)."""
    state: dict[str, torch.Tensor] = {}
    decoded = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in keys:
            if key.endswith(SKIPPED):  # a marker or scale, read with its weight below; ComfyUI's kernel hint
                continue
            tensor = handle.get_tensor(key)
            base = key.removesuffix(".weight")
            if key.endswith(".weight") and f"{base}.comfy_quant" in keys:
                config = parse_comfy_quant_marker(handle.get_tensor(f"{base}.comfy_quant"))
                if config.get("format") != "int8_tensorwise" or config.get("convrot") is not True:
                    raise ValueError(f"{base}.comfy_quant is not an INT8 ConvRot marker: {config}")
                scale = handle.get_tensor(f"{base}.weight_scale")
                tensor = dequantize_weight(tensor, scale, int(config["convrot_groupsize"]), dtype)
                decoded += 1
            state[key] = tensor
    return state, decoded


def _refuse_edit_turbo(path: Path, quantized: bool) -> None:
    """Edit Turbo is step-distilled: BRIA says it is not a fine-tuning base. Its INT8 repack is refused, a bf16 file
    (which may have been renamed) is only warned about."""
    if "edit_turbo" not in path.name.lower():
        return
    if quantized:
        raise ValueError(
            f"{path.name} is the INT8 repack of Fibo 2.0 Edit Turbo, which is distilled and not a fine-tuning "
            "base. Train on fibo2_dev_bf16.safetensors (or fibo2_dev_int8_convrot.safetensors)."
        )
    logger.warning("%s looks like Fibo 2.0 Edit Turbo, which is distilled and not a fine-tuning base.", path.name)


def _read_state(path: Path, quantized: bool, int8_load: str, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], int]:
    """(the file's tensors as the model takes them, how many Linears are int8 in them)."""
    if not quantized:
        return load_file(str(path)), 0
    if int8_load == "runtime":
        return load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    return _decoded_state(path, dtype)


def load_fibo2_transformer(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    int8_load: str = "decode",
    grad_precision: str = "bf16",
) -> tuple[Fibo2Transformer, dict[str, Any]]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fibo 2.0 transformer not found at {path}")
    if int8_load not in INT8_LOADS:
        raise ValueError(f"model.int8_load must be one of {INT8_LOADS}, got {int8_load!r}")
    quantized = is_quantized(path)
    _refuse_edit_turbo(path, quantized)
    logger.info("Loading Fibo 2.0 DiT from %s%s...", path, f" (INT8 ConvRot, {int8_load})" if quantized else "")
    runtime = quantized and int8_load == "runtime"
    raw, layers = _read_state(path, quantized, int8_load, dtype)
    state = strip_container_prefix(raw)
    config = Fibo2Config.from_state_dict(state)

    with init_empty_weights(include_buffers=False):
        model = Fibo2Transformer(config)
    if runtime:
        prepare_int8_convrot_modules(model, state)
    model.load_state_dict(state, strict=True, assign=True)
    if runtime:
        enable_int8_convrot(model, grad_precision=grad_precision)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.requires_grad_(False)
    model.to(device=device)
    model.compute_dtype = dtype
    model.eval()
    info = {
        "file": path.name,
        "format": "int8_convrot"
        if quantized
        else str(state["layers.0.attention.qkv.weight"].dtype).replace("torch.", ""),
        "dim": config.dim,
        "layers": config.n_layers,
        "injection_layers": list(config.injection_layers),
    }
    if quantized:
        info.update(int8_load=int8_load, int8_layers=layers)
    logger.info("Fibo 2.0 DiT ready: %s", info)
    return model, info
