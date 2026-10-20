"""The MiniMax H3 LoRA trainer's model: the INT8 ConvRot transformer base with the LoRA on it."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights

from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig

logger = logging.getLogger(__name__)


def load_transformer(
    config: dict[str, Any], device: torch.device, run_metadata: dict[str, object], say: Callable[[str], None]
) -> MiniMaxH3Transformer:
    """The transformer from ComfyUI's INT8 ConvRot file, frozen, gradient-checkpointed, on ``device``."""
    int8_state_dict, num_quant_layers = load_comfy_int8_convrot_state_dict(
        config.get("model", {})["transformer_path"], device=torch.device("cpu")
    )
    logger.info("Parsed %d INT8 ConvRot quantized layers", num_quant_layers)

    if "adaln_t_table" in int8_state_dict:
        table = int8_state_dict["adaln_t_table"]
        table_size, embed_dim = table.shape[0], table.shape[1]
        logger.info("Detected pruned AdaLN table: size=%d, embed_dim=%d", table_size, embed_dim)
        transformer_cfg = MiniMaxH3TransformerConfig(adaln_t_table_size=table_size, time_embed_dim=embed_dim)
    else:
        transformer_cfg = MiniMaxH3TransformerConfig()

    say("Instantiating Transformer structure on GPU...")
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Transformer(transformer_cfg)

    prepare_int8_convrot_modules(transformer, int8_state_dict)
    info = transformer.load_state_dict(int8_state_dict, strict=False, assign=True)
    # Everything is built under init_empty_weights, so an unloaded tensor stays on the
    # meta device. rope.inv_freq and adaln_t_table in particular carry checkpoint values
    # that cannot be recomputed -- silently keeping a placeholder would misplace every
    # RoPE frequency, so fail loudly instead.
    if info.missing_keys:
        raise ValueError(
            f"Transformer checkpoint is missing {len(info.missing_keys)} tensor(s); examples: {info.missing_keys[:5]}"
        )
    if info.unexpected_keys:
        raise ValueError(
            f"Transformer checkpoint has {len(info.unexpected_keys)} tensor(s) the model does "
            f"not expect; examples: {info.unexpected_keys[:5]}"
        )
    grad_precision = config.get("train", {}).get("int8_grad_precision")
    enable_int8_convrot(transformer, grad_precision=grad_precision)
    run_metadata["int8_grad_precision"] = grad_precision or "bf16"

    transformer.enable_gradient_checkpointing()
    transformer.requires_grad_(False)
    transformer.to(device)
    return transformer


def add_lora(
    transformer: MiniMaxH3Transformer,
    config: dict[str, Any],
    on: tuple[torch.device, torch.dtype],
    run_metadata: dict[str, object],
) -> tuple[MiniMaxH3LoRANetwork, torch.dtype]:
    """The LoRA network over ``transformer`` (and a training adapter, if set): (network, the adapters' dtype)."""
    device, dtype = on
    model_config, lora_config = config.get("model", {}), config.get("lora", {})
    network = MiniMaxH3LoRANetwork(
        transformer=transformer,
        rank=lora_config.get("rank", 16),
        alpha=lora_config.get("alpha", 16.0),
        prefix=lora_config.get("prefix", "diffusion_model."),
        target_modules=lora_config.get("target_modules", ["qkv_proj", "out_proj", "fc1", "fc2"]),
        include_adaln=bool(lora_config.get("include_adaln", False)),
        include_token_refiner=bool(lora_config.get("include_token_refiner", True)),
        block_range=lora_config.get("block_range"),
    )
    # Adapters train in fp32 by default. bf16 carries ~8 mantissa bits (~0.4% relative
    # precision), and a LoRA update is small next to the weight it lands on, so in bf16 a
    # large share of every step rounds away -- the subject trains "a bit" and then stops
    # sharpening. Cast only the adapters: the base keeps its checkpoint dtypes, including
    # the tensors H3 holds in fp32.
    lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
    network.set_lora_dtype(device, lora_dtype)
    adapter_path = model_config.get("training_adapter_path") or lora_config.get("training_adapter_path")
    if adapter_path:
        attached = network.load_assistant_adapter(
            adapter_path,
            multiplier=float(lora_config.get("training_adapter_multiplier", 1.0)),
            device=device,
            dtype=dtype,
            adaln_grid_path=model_config.get("training_adapter_adaln_grid"),
        )
        run_metadata["training_adapter"] = Path(str(adapter_path)).name
        run_metadata["training_adapter_modules"] = attached

    trainable = sum(p.numel() for p in network.trainable_parameters())
    run_metadata["lora_dtype"] = str(lora_dtype).replace("torch.", "")
    run_metadata["trainable_parameters"] = trainable
    run_metadata["lora_layers"] = len(network.lora_layers)
    logger.info(
        "LoRA adapters: %.1fM trainable parameters in %s (base left at checkpoint dtypes)", trainable / 1e6, lora_dtype
    )
    return network, lora_dtype
