"""
Loading ComfyUI's FLUX.3 [klein] single-file DiT checkpoints.

Two storage formats:
- ``flux3_image_klein_int8_convrot``: every block Linear is int8 with a per-row scale and a
  Hadamard-rotated input (ComfyUI's INT8 ConvRot). It stays int8 on the GPU and runs through
  the int8 kernels shared with MiniMax H3 and Qwen-Image 2.1. The LoRA learns against exactly
  the weights ComfyUI runs.
- ``flux3_image_klein_bf16``: plain weights.

The step-distilled ``flux3_image_klein_sd_*`` files carry a ``ref_time_in`` flow-map embedder.
They are refused: BFL's model card says to fine-tune the base model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from maltrainer.models.flux3.model import Flux3Config, Flux3Transformer
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)

logger = logging.getLogger(__name__)

CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def is_int8_convrot(path: Union[str, Path]) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(key.endswith(".comfy_quant") for key in handle.keys())


def _strip_container_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for prefix in CONTAINER_PREFIXES:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_flux3_transformer(
    path: Union[str, Path],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    grad_precision: str = "bf16",
) -> Tuple[Flux3Transformer, dict]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FLUX.3 transformer not found at {path}")

    quantized = is_int8_convrot(path)
    logger.info("Loading FLUX.3 DiT from %s (%s)...", path, "INT8 ConvRot" if quantized else "plain")
    if quantized:
        state_dict, num_quant = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        state_dict, num_quant = load_file(str(path)), 0
    state_dict = _strip_container_prefix(state_dict)

    config = Flux3Config.from_state_dict(state_dict)
    if config.flow_map:
        raise ValueError(
            f"{path.name} is the step-distilled FLUX.3 checkpoint (it has a ref_time_in flow-map "
            "embedder). Train on the base model: flux3_image_klein_int8_convrot or _bf16, "
            "without _sd_ in the name."
        )
    # The image_cond (reference image) mode and its early blocks stay frozen and unused.

    with init_empty_weights(include_buffers=False):
        model = Flux3Transformer(config)
    if num_quant:
        prepare_int8_convrot_modules(model, state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)
    if num_quant:
        enable_int8_convrot(model, grad_precision=grad_precision)

    # Cast only the float weights. The int8 per-row scales stay fp32, as in ComfyUI.
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    model.requires_grad_(False)
    model.to(device=device)
    model.compute_dtype = dtype
    model.eval()

    stored = state_dict["single_blocks.0.attn_out.weight"].dtype
    info = {
        "file": path.name,
        "format": "int8_convrot" if quantized else str(stored).replace("torch.", ""),
        "int8_layers": num_quant,
        "hidden_size": config.hidden_size,
        "depth": config.depth,
        "depth_single_blocks": config.depth_single_blocks,
        "fused_qkv": config.fused_qkv,
    }
    logger.info("FLUX.3 DiT ready: %s", info)
    return model, info
