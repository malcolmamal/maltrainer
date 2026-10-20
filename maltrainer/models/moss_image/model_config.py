"""
MOSS-Image DiT config, read back off a checkpoint's state dict (see ``model``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MOSSImageConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    fused_qkv: bool = False

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> MOSSImageConfig:
        """comfy/model_detection.py's MOSS-Image branch."""
        keys = state_dict.keys()
        if "shared_block_modulation.weight" not in keys or "layers.0.proj.ffn_up_gate.weight" not in keys:
            raise ValueError("Not a MOSS-Image DiT checkpoint (no shared_block_modulation / layers.0.proj.ffn_up_gate)")
        dim = state_dict["shared_block_modulation.weight"].shape[0] // 6
        return cls(
            dim=dim,
            ffn_dim=state_dict["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
            num_heads=dim // 128,
            num_layers=len({k.split(".", 2)[1] for k in keys if k.startswith("layers.")}),
            text_dim=state_dict["text_embedder.proj.weight"].shape[1],
            fused_qkv="layers.0.proj.qkv.weight" in keys,
        )
