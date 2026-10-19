"""Synthetic ComfyUI-layout FLUX.3 checkpoints written to disk for the loader tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

from maltrainer.models.flux3.model import Flux3Config
from tests.maltrainer.models.flux3.tiny_model import TINY, tiny_transformer

GROUP_SIZE = 16
BLOCK_LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
BLOCK_SECTIONS = ("txt_mode_blocks.", "content_mode_blocks.", "single_blocks.")


def marker(**fields: object) -> torch.Tensor:
    return torch.tensor(list(json.dumps(fields).encode("utf-8")), dtype=torch.uint8)


def plain_state(config: Flux3Config = TINY, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """The module state dict, with the QK-norm weights spelt ``*_norm.scale`` as BFL's files do."""
    state = {}
    for key, value in tiny_transformer(config).state_dict().items():
        key = key[: -len(".weight")] + ".scale" if key.endswith("_norm.weight") else key
        state[key] = value.detach().clone().to(dtype)
    return state


def is_block_linear(key: str) -> bool:
    return key.startswith(BLOCK_SECTIONS) and key.rsplit(".", 2)[-2] in BLOCK_LINEARS and key.endswith(".weight")


def int8_state(config: Flux3Config = TINY) -> Dict[str, torch.Tensor]:
    """Every block Linear int8 with a per-row scale and a ConvRot marker; the rest bf16."""
    generator = torch.Generator().manual_seed(3)
    state = {key: value.to(torch.bfloat16) for key, value in plain_state(config).items()}
    for key in [k for k in state if is_block_linear(k)]:
        base = key[: -len(".weight")]
        rows, cols = state[key].shape
        state[key] = torch.randint(-100, 100, (rows, cols), generator=generator, dtype=torch.int8)
        state[f"{base}.weight_scale"] = torch.full((rows,), 0.01)
        state[f"{base}.comfy_quant"] = marker(format="int8_tensorwise", convrot=True, convrot_groupsize=GROUP_SIZE)
    return state


def write(path: Path, state: Dict[str, torch.Tensor], prefix: str = "") -> Path:
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, str(path))
    return path
