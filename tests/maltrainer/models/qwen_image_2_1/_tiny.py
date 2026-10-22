from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.model import QwenImage21Transformer

TINY_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "dim": 16,
    "heads": 2,
    "dim_head": 8,
    "mlp_ratio": 2,
    "context_in_dim": 6,
    "axes_dims_rope": (2, 2, 4),
}


def tiny_transformer(seed: int = 0, **overrides: object) -> QwenImage21Transformer:
    torch.manual_seed(seed)
    return QwenImage21Transformer(**{**TINY_CONFIG, **overrides})


def tiny_inputs(batch: int = 2, h: int = 2, w: int = 3, text_len: int = 3, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(batch, h * w, TINY_CONFIG["in_channels"], generator=generator)
    text = torch.randn(batch, text_len, TINY_CONFIG["context_in_dim"], generator=generator)
    timestep = torch.rand(batch, generator=generator)
    return latents, text, timestep


def int8_checkpoint(
    path: Path, state: dict[str, torch.Tensor], quantized: dict[str, int], seed: int = 0
) -> dict[str, torch.Tensor]:
    """Write ``state`` as a Comfy INT8 ConvRot file; ``quantized`` maps a linear name to its group size.

    Returns the int8 weights and fp32 scales that were stored, keyed by linear name.
    """
    generator = torch.Generator().manual_seed(seed)
    stored: dict[str, torch.Tensor] = {}
    tensors = {key: value.clone() for key, value in state.items()}
    for name, group in quantized.items():
        shape = tensors.pop(f"{name}.weight").shape
        weight = torch.randint(-100, 100, shape, generator=generator, dtype=torch.int8)
        scale = torch.rand(shape[0], generator=generator) * 0.01 + 0.005
        marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
        tensors[f"{name}.weight"] = weight
        tensors[f"{name}.weight_scale"] = scale
        tensors[f"{name}.comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
        stored[name] = weight
    save_file(tensors, str(path))
    return stored
