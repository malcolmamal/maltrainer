"""A tiny MOSS-Image DiT and the tensors a checkpoint file holds, built from explicit small sizes.

dim 128 is the smallest the loaders accept: ``MOSSImageConfig.from_state_dict`` reads the head
count as ``dim // 128``. Everything else is shrunk to a few channels.
"""

from __future__ import annotations

import json
from typing import Dict

import torch

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.moss_image.model import MOSSImageConfig, MOSSImageTransformer

TEXT_DIM = 8
GROUP_SIZE = 16
CPU = torch.device("cpu")


def tiny_config(fused_qkv: bool = False, num_layers: int = 2) -> MOSSImageConfig:
    return MOSSImageConfig(
        dim=128, ffn_dim=16, num_heads=1, num_layers=num_layers, text_dim=TEXT_DIM, fused_qkv=fused_qkv
    )


def tiny_model(fused_qkv: bool = False, seed: int = 0, num_layers: int = 2) -> MOSSImageTransformer:
    torch.manual_seed(seed)
    model = MOSSImageTransformer(tiny_config(fused_qkv, num_layers))
    model.compute_dtype = torch.float32
    return model


def json_tensor(value: object) -> torch.Tensor:
    return torch.tensor(list(json.dumps(value).encode("utf-8")), dtype=torch.uint8)


def int8_tensors(model: MOSSImageTransformer) -> Dict[str, torch.Tensor]:
    """The block Linears as a ComfyUI INT8 ConvRot file stores them: Hadamard-rotated int8 rows, a per-row scale, a marker."""
    marker = json_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE})
    out: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if not (key.startswith("layers.") and ".proj." in key and key.endswith(".weight")):
            out[key] = value
            continue
        base = key[: -len(".weight")]
        rotated = rotate_activation(value, GROUP_SIZE)
        scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
        out[key] = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
        out[f"{base}.weight_scale"] = scale.reshape(-1)
        out[f"{base}.comfy_quant"] = marker
    return out
