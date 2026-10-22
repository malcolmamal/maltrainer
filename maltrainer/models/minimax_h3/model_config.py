"""Config, token tags and output container of the MiniMax H3 transformer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

MINIMAX_H3_MODALITY_COUNT = 3


class MiniMaxH3TokenTag(IntEnum):
    VIDEO = 0
    TEXT = 1
    AUDIO = 2


@dataclass(frozen=True)
class MiniMaxH3TransformerConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_t_table_size: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MiniMaxH3TransformerConfig:
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


@dataclass
class MiniMaxH3TransformerOutput:
    video: torch.Tensor
    audio: torch.Tensor
