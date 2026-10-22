"""The Z-Image LoRA trainer's model: the diffusers transformer, text encoder, tokenizer and VAE, with an
optional training adapter (Z-Image Turbo's) merged into the transformer's weights first."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def resolve_hf_path(path_value: str) -> str:
    """A local file as it is; ``owner/repo/file...`` downloaded from the Hub."""
    if Path(path_value).exists():
        return path_value
    parts = path_value.split("/")
    if len(parts) < 3:
        raise ValueError(f"Adapter path {path_value} is not a valid local path or hub path.")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return hf_hub_download(repo_id=repo_id, filename=filename)


def merge_lora(transformer: torch.nn.Module, lora_state_dict: dict[str, Any], merge_weight: float = 1.0) -> int:
    """Add each LoRA pair (B @ A * alpha / rank * weight) into its Linear; returns how many merged."""
    module_map = dict(transformer.named_modules())
    merged = 0
    for key, lora_a in lora_state_dict.items():
        if not key.endswith(".lora_A.weight"):
            continue
        base_key = key[: -len(".lora_A.weight")]
        lora_b_key = f"{base_key}.lora_B.weight"
        if lora_b_key not in lora_state_dict:
            continue
        module_name = base_key.removeprefix("diffusion_model.").removeprefix("transformer.")
        module = module_map.get(module_name)
        if module is None or not isinstance(module, torch.nn.Linear):
            continue

        rank = int(lora_a.shape[0])
        alpha = float(lora_state_dict.get(f"{base_key}.alpha", rank))
        scale = (alpha / rank) * float(merge_weight)
        with torch.no_grad():
            a_dev = lora_a.to(device=module.weight.device, dtype=module.weight.dtype)
            b_dev = lora_state_dict[lora_b_key].to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.add_(torch.matmul(b_dev, a_dev) * scale)
        merged += 1
    return merged


def load_components(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype, text_encoder_cls: Any = None
) -> tuple[Any, Any, Any, Any]:
    """(transformer, text encoder, tokenizer, VAE), frozen and on ``device``; the VAE in float32.

    The LoRA trainer's default loads the extras from ``model.extras_name_or_path`` and merges
    ``model.assistant_lora_path``; a ``text_encoder_cls`` (the embedding trainer's) loads everything from
    ``model.name_or_path`` and merges nothing.
    """
    model_config = config.get("model", {})
    model_path = model_config["name_or_path"]
    lora = text_encoder_cls is None
    base_model_path = model_config.get("extras_name_or_path", model_path) if lora else model_path
    assistant_lora_path = model_config.get("assistant_lora_path", None) if lora else None

    transformer = ZImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    transformer.enable_gradient_checkpointing()
    if assistant_lora_path:  # Z-Image Turbo's training adapter
        merged = merge_lora(transformer, load_file(resolve_hf_path(str(assistant_lora_path))), merge_weight=1.0)
        logger.info("Merged %d adapter layers from %s", merged, assistant_lora_path)

    text_encoder: Any = cast(Any, text_encoder_cls or AutoModelForCausalLM).from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer: Any = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    vae = cast(Any, AutoencoderKL).from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)

    for model in (transformer, text_encoder, vae):
        model.requires_grad_(False)
    for model in (transformer, text_encoder, vae):
        model.to(device)
    return transformer, text_encoder, tokenizer, vae
