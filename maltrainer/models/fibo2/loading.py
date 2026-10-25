"""
Loading ComfyUI's Fibo 2.0 single-file DiT (``fibo2_dev_bf16.safetensors``).

The file is the plain bf16 DiT. ComfyUI's INT8 ConvRot repack (``fibo2_edit_turbo_int8_convrot``) is the
step-distilled Edit Turbo model, which BRIA says is not a fine-tuning base, so it is refused with that reason.
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


def load_fibo2_transformer(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16
) -> tuple[Fibo2Transformer, dict[str, Any]]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fibo 2.0 transformer not found at {path}")
    if is_quantized(path):
        raise ValueError(
            f"{path.name} is an INT8 ConvRot file. ComfyUI ships those only for Fibo 2.0 Edit Turbo, which is "
            "distilled and not a fine-tuning base. Train on fibo2_dev_bf16.safetensors."
        )
    logger.info("Loading Fibo 2.0 DiT from %s...", path)
    state = strip_container_prefix(load_file(str(path)))
    config = Fibo2Config.from_state_dict(state)

    with init_empty_weights(include_buffers=False):
        model = Fibo2Transformer(config)
    model.load_state_dict(state, strict=True, assign=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)
    model.requires_grad_(False)
    model.to(device=device)
    model.compute_dtype = dtype
    model.eval()
    info = {
        "file": path.name,
        "format": str(state["layers.0.attention.qkv.weight"].dtype).replace("torch.", ""),
        "dim": config.dim,
        "layers": config.n_layers,
        "injection_layers": list(config.injection_layers),
    }
    logger.info("Fibo 2.0 DiT ready: %s", info)
    return model, info
