"""A tiny Qwen3.5 text model and the files that hold it.

``Qwen35TextConfig.from_state_dict`` assumes the linear-attention key head size is 128, so the
tiny model keeps that and shrinks everything else (one key head, two value heads of 8).
"""

from __future__ import annotations

import torch

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.moss_image.text_encoder import Qwen35TextConfig, Qwen35TextModel

from .tiny import json_tensor

HIDDEN = 16
GROUP = 4

TINY = Qwen35TextConfig(
    vocab_size=20,
    hidden_size=HIDDEN,
    intermediate_size=24,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_value_head_dim=8,
    full_attention_layers=(3,),
)


def tiny_text_model(seed: int = 0) -> Qwen35TextModel:
    torch.manual_seed(seed)
    model = Qwen35TextModel(TINY)
    for param in model.parameters():
        param.data.normal_(std=0.3)
    return model


def _quantize(value: torch.Tensor) -> tuple:
    rotated = rotate_activation(value, GROUP)
    scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (rotated / scale).round().clamp(-127, 127).to(torch.int8), scale.reshape(-1)


def text_int8_tensors(model: Qwen35TextModel) -> dict[str, torch.Tensor]:
    """Every Linear and the embedding table as an INT8 ConvRot file stores them."""
    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
    linears = {name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)} | {"embed_tokens"}
    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        base = key[: -len(".weight")]
        if not key.endswith(".weight") or base not in linears:
            out[key] = value
            continue
        out[key], out[f"{base}.weight_scale"] = _quantize(value)
        out[f"{base}.comfy_quant"] = json_tensor(marker)
    return out
