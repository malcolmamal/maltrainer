"""The Qwen-Image 2.1 LoRA trainer's model: the transformer base (INT8 ConvRot or BF16) with the LoRA on it."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork
from maltrainer.models.qwen_image_2_1.model import load_qwen_image_2_1_transformer

logger = logging.getLogger(__name__)


def load_lora_model(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype, run_metadata: dict[str, object]
) -> tuple[Any, QwenImage21LoRANetwork, torch.dtype]:
    """(transformer, its LoRA network, the adapters' dtype); what was loaded goes into ``run_metadata``."""
    logger.info("Loading Transformer base and LoRA network...")
    model_config = config.get("model", {})
    grad_precision = config.get("train", {}).get("int8_grad_precision", "bf16")
    transformer = load_qwen_image_2_1_transformer(
        model_config["transformer_path"], device=device, dtype=dtype, grad_precision=grad_precision
    )
    run_metadata["int8_grad_precision"] = grad_precision

    lora_config = config.get("lora", {})
    network = QwenImage21LoRANetwork(
        transformer=transformer,
        rank=lora_config.get("rank", 16),
        alpha=lora_config.get("alpha", 16.0),
        prefix=lora_config.get("prefix", "diffusion_model."),
        target_modules=lora_config.get("target_modules", None),
    )
    if not network.lora_layers:
        raise ValueError(f"No Qwen-Image 2.1 module matched lora.target_modules {lora_config.get('target_modules')}")
    lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
    network.set_lora_dtype(device, lora_dtype)

    adapter_path = model_config.get("training_adapter_path") or lora_config.get("training_adapter_path")
    if adapter_path:
        multiplier = float(lora_config.get("training_adapter_multiplier", 1.0))
        attached = network.load_assistant_adapter(adapter_path, multiplier=multiplier, device=device, dtype=dtype)
        run_metadata["training_adapter"] = Path(str(adapter_path)).name
        run_metadata["training_adapter_modules"] = attached

    trainable = sum(p.numel() for p in network.trainable_parameters())
    run_metadata["lora_dtype"] = str(lora_dtype).replace("torch.", "")
    run_metadata["trainable_parameters"] = trainable
    run_metadata["lora_layers"] = len(network.lora_layers)
    logger.info(
        "LoRA adapters: %.1fM trainable parameters in %s (base left at checkpoint dtypes)", trainable / 1e6, lora_dtype
    )
    return transformer, network, lora_dtype
