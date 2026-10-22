"""
Loading ComfyUI's MOSS-Image single-file DiT checkpoints (Comfy-Org/MOSS-Image).

Two storage formats:
- ``moss_image_base_bf16``: plain weights.
- ``moss_image_base_int8_convrot``: block Linears int8 with a per-row scale and a Hadamard-rotated
  input (ComfyUI's INT8 ConvRot), run through the int8 kernels shared with MiniMax H3, Qwen-Image
  2.1 and FLUX.3.

Train the base model. The ``_sft_`` files are OpenMOSS's supervised fine-tune of it; they load
too, and the trainer warns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.moss_image.model import MOSSImageConfig, MOSSImageTransformer

logger = logging.getLogger(__name__)

CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _strip_container_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in CONTAINER_PREFIXES:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _to_module_names(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Drop what is not a weight, as ComfyUI does on load.

    - ``rope.bands``: the training code's RoPE frequency buffer. It is recomputed
      (MOSSImage.process_unet_state_dict pops it).
    - ``<block>.comfy_attention.config``: a JSON hint naming the attention kernel ComfyUI's
      inference may use. Training runs plain SDPA, so it is dropped and only reported.
    """
    out: dict[str, torch.Tensor] = {}
    attention: dict[str, int] = {}
    for key, value in state_dict.items():
        if key == "rope.bands":
            continue
        if key.endswith(".comfy_attention.config"):
            try:
                config = json.loads(bytes(value.flatten().tolist()).decode("utf-8"))
                configs = config if isinstance(config, list) else [config]
                methods = ",".join(str(c.get("attention")) for c in configs)
            except (ValueError, AttributeError):
                methods = "unreadable"
            attention[methods] = attention.get(methods, 0) + 1
            continue
        out[key] = value
    return out, {method: f"{count} blocks" for method, count in attention.items()}


def load_moss_image_transformer(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    grad_precision: str = "bf16",
) -> tuple[MOSSImageTransformer, dict[str, Any]]:
    """The frozen DiT on ``device``, plus a small summary for the checkpoint metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MOSS-Image transformer not found at {path}")
    if "_sft" in path.name:
        logger.warning(
            "%s is the SFT (supervised fine-tune) checkpoint; LoRAs are normally trained on the base model.", path.name
        )

    quantized = is_int8_convrot(path)
    logger.info("Loading MOSS-Image DiT from %s (%s)...", path, "INT8 ConvRot" if quantized else "plain")
    if quantized:
        state_dict, num_quant = load_comfy_int8_convrot_state_dict(str(path), device=torch.device("cpu"))
    else:
        state_dict, num_quant = load_file(str(path)), 0
    state_dict, attention_hints = _to_module_names(_strip_container_prefix(state_dict))
    if attention_hints:
        logger.info("Checkpoint attention hints for ComfyUI inference (not used in training): %s", attention_hints)

    config = MOSSImageConfig.from_state_dict(state_dict)
    with init_empty_weights(include_buffers=False):
        model = MOSSImageTransformer(config)
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

    stored = state_dict["layers.0.proj.o.weight"].dtype
    info = {
        "file": path.name,
        "format": "int8_convrot" if quantized else str(stored).replace("torch.", ""),
        "int8_layers": num_quant,
        "dim": config.dim,
        "num_layers": config.num_layers,
        "fused_qkv": config.fused_qkv,
        "inference_attention": attention_hints,
    }
    logger.info("MOSS-Image DiT ready: %s", info)
    return model, info
