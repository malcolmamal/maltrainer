"""The MiniMax H3 Image LoRA trainer's model: the INT8 ConvRot transformer, its noise levels, and the LoRA on it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3.model_config import MiniMaxH3TransformerConfig
from maltrainer.models.minimax_h3_image.grid import file_pdd_sigmas
from maltrainer.models.minimax_h3_image.model import MiniMaxH3ImageTransformer
from maltrainer.trainers.minimax_h3.model import add_lora, load_transformer


def load_image_transformer(
    config: dict[str, Any],
    on: tuple[torch.device, torch.dtype],
    run_metadata: dict[str, object],
    say: Callable[[str], None],
) -> tuple[MiniMaxH3ImageTransformer, MiniMaxH3LoRANetwork, torch.dtype, tuple[float, ...]]:
    """(transformer, LoRA network, the adapters' dtype, the checkpoint's sigma grid including the final 0)."""
    sigmas = file_pdd_sigmas(config["model"]["transformer_path"])
    levels = len(sigmas) - 1

    def build(transformer_config: MiniMaxH3TransformerConfig) -> MiniMaxH3ImageTransformer:
        return MiniMaxH3ImageTransformer(transformer_config, levels)

    transformer = load_transformer(config, on[0], run_metadata, say, build)
    run_metadata["noise_levels"] = list(sigmas[:-1])
    network, lora_dtype = add_lora(transformer, config, on, run_metadata)
    return transformer, network, lora_dtype, sigmas
