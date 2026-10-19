"""A tiny Qwen3.5 text model, and its weights as the checkpoint files the loader reads.

Sizes are the smallest ``Qwen35TextConfig.from_state_dict`` can read back: it assumes a linear
key head dim of 128, so the linear attention has one key head of 128 and two value heads of 4.
Layers 0-2 are Gated DeltaNet, layer 3 is gated full attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.moss_image.text_encoder import Qwen35TextConfig, Qwen35TextModel

from .tiny import json_tensor

VOCAB = 32
HIDDEN = 16
GROUP = 4
MLP_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def tiny_text_config(layers: int = 4) -> Qwen35TextConfig:
    return Qwen35TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=4,
        conv_kernel_size=4,
        full_attention_layers=tuple(i for i in range(layers) if (i + 1) % 4 == 0),
    )


def randomize(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Random weights everywhere (the real init zeroes the additive norms and A_log)."""
    torch.manual_seed(seed)
    for param in module.parameters():
        param.data = torch.randn_like(param) * 0.3
    return module.eval()


def tiny_text_model(seed: int = 0, layers: int = 4) -> Qwen35TextModel:
    return randomize(Qwen35TextModel(tiny_text_config(layers)), seed)


def quantized_tensors(model: Qwen35TextModel) -> Dict[str, torch.Tensor]:
    """The MLP Linears and the embedding as a ComfyUI INT8 ConvRot file stores them."""
    marker = json_tensor({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP})
    out: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.endswith(MLP_SUFFIXES) or key == "embed_tokens.weight":
            out.update(_quantize(key, value, marker))
        else:
            out[key] = value
    return out


def _quantize(key: str, value: torch.Tensor, marker: torch.Tensor) -> Dict[str, torch.Tensor]:
    base = key[: -len(".weight")]
    rotated = rotate_activation(value, GROUP)
    scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return {
        key: (rotated / scale).round().clamp(-127, 127).to(torch.int8),
        f"{base}.weight_scale": scale.reshape(-1),
        f"{base}.comfy_quant": marker.clone(),
    }


def write_text_checkpoint(tensors: Dict[str, torch.Tensor], path: Path, prefix: str = "") -> Path:
    save_file({prefix + k: v.detach().contiguous() for k, v in tensors.items()}, str(path))
    return path
