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

import json
import logging
from pathlib import Path

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


def is_int8_convrot(path: str | Path) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(key.endswith(".comfy_quant") for key in handle.keys())


def _strip_container_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in CONTAINER_PREFIXES:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _to_module_names(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """BFL's key spellings -> the module names, as ComfyUI's Flux.process_unet_state_dict does.

    - QK-norm weights are stored as ``*_norm.scale``.
    - ``<block>.comfy_attention.config`` is a JSON hint naming the attention kernel ComfyUI's
      inference may use (comfy-kitchen int8 / SOL). It holds no weights; training runs plain
      SDPA, so it is dropped and only reported.
    """
    out, attention = {}, {}
    for key, value in state_dict.items():
        if key.endswith(".comfy_attention.config"):
            try:
                config = json.loads(bytes(value.flatten().tolist()).decode("utf-8"))
                configs = config if isinstance(config, list) else [config]
                methods = ",".join(str(c.get("attention")) for c in configs)
            except (ValueError, AttributeError):
                methods = "unreadable"
            attention[methods] = attention.get(methods, 0) + 1
            continue
        name = key[: -len(".scale")] + ".weight" if key.endswith("_norm.scale") else key
        out[name] = value
    return out, {method: f"{count} blocks" for method, count in attention.items()}


def load_flux3_transformer(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    grad_precision: str = "bf16",
) -> tuple[Flux3Transformer, dict]:
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
    state_dict, attention_hints = _to_module_names(_strip_container_prefix(state_dict))
    if attention_hints:
        logger.info("Checkpoint attention hints for ComfyUI inference (not used in training): %s", attention_hints)

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
        "inference_attention": attention_hints,
    }
    logger.info("FLUX.3 DiT ready: %s", info)
    return model, info
