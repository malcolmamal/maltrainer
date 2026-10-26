"""
Loading the Visage-Image DiT from ComfyUI's single file (``diffusion_models/Visage/visage-image.safetensors``): the
bf16 transformer under the module names of the Hub repo's ``transformer/model.safetensors``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from maltrainer.models.visage.config import VisageConfig
from maltrainer.models.visage.model import VisageTransformer

logger = logging.getLogger(__name__)

CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def strip_container_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in CONTAINER_PREFIXES:
        if state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_visage_transformer(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16
) -> tuple[VisageTransformer, dict[str, Any]]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Visage-Image transformer not found at {path}")
    logger.info("Loading Visage-Image DiT from %s...", path)
    state = strip_container_prefix(load_file(str(path)))
    config = VisageConfig.from_state_dict(state)
    with init_empty_weights(include_buffers=False):
        model = VisageTransformer(config)
    model.load_state_dict(state, strict=True, assign=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)
    model.requires_grad_(False)
    model.to(device=device)
    model.eval()
    info = {
        "file": path.name,
        "format": str(state["blocks.0.attn.qkv.weight"].dtype).replace("torch.", ""),
        "width": config.width,
        "depth": config.depth,
        "latent_channels": config.in_channels,
    }
    logger.info("Visage-Image DiT ready: %s", info)
    return model, info
