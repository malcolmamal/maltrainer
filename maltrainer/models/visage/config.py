"""The Visage-Image DiT's configuration. The single file carries no config, so it is read off the tensor shapes."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

HEAD_DIM = 64
DEFAULT_BUCKETS = (32, 64, 96, 128)  # caption lengths the pad embeddings fill up to
REQUIRED = (
    "text_proj.weight",
    "text_pad_embed",
    "text_registers",
    "patch_embed.weight",
    "time_embed.1.weight",
    "blocks.0.mlp.fc2.weight",
    "out_proj.weight",
)


@dataclass(frozen=True)
class VisageConfig:
    in_channels: int = 96  # VAE latent channels
    patch_size: int = 1
    width: int = 768
    depth: int = 18
    mlp_ratio: int = 4
    text_dim: int = 1024  # Qwen3-0.6B hidden size
    noise_embed_dims: int = 256
    time_min_freq: float = 0.01
    time_max_freq: float = 100.0
    text_length_buckets: tuple[int, ...] = DEFAULT_BUCKETS
    n_text_registers: int = 16

    @property
    def heads(self) -> int:
        return self.width // HEAD_DIM

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> VisageConfig:
        """The configuration of the DiT a state dict (module names, no container prefix) holds."""
        missing = [key for key in REQUIRED if key not in state]
        if missing:
            raise ValueError(f"not a Visage-Image DiT checkpoint (missing {', '.join(missing)})")
        width, text_dim = state["text_proj.weight"].shape
        longest = state["text_pad_embed"].shape[0]
        depth = 1 + max(int(m.group(1)) for key in state if (m := re.match(r"blocks\.(\d+)\.", key)))
        return cls(
            in_channels=state["patch_embed.weight"].shape[1],
            patch_size=state["patch_embed.weight"].shape[2],
            width=width,
            depth=depth,
            mlp_ratio=state["blocks.0.mlp.fc2.weight"].shape[1] // width,
            text_dim=text_dim,
            noise_embed_dims=state["time_embed.1.weight"].shape[1],
            text_length_buckets=(*(b for b in DEFAULT_BUCKETS if b < longest), longest),
            n_text_registers=state["text_registers"].shape[0],
        )
