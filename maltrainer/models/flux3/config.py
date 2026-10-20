"""FLUX.3 DiT configuration, read off a checkpoint the way comfy/model_detection.py does."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Flux3Config:
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    fused_qkv: bool = False
    flow_map: bool = False
    mlp_ratio: float = 3.0

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> Flux3Config:
        """comfy/model_detection.py's FLUX.3 branch."""
        keys = state_dict.keys()
        if (
            "content_mode_blocks.image.0.attn_out.weight" not in keys
            or "single_stream_modulations.txt.lin.weight" not in keys
        ):
            raise ValueError("Not a FLUX.3 DiT checkpoint (no content_mode_blocks / single_stream_modulations)")
        hidden = state_dict["content_mode_blocks.image.0.attn_out.weight"].shape[0]

        def count(prefix: str) -> int:
            return len({k[len(prefix) :].split(".", 1)[0] for k in keys if k.startswith(prefix)})

        return cls(
            hidden_size=hidden,
            num_heads=hidden // 128,
            depth=count("txt_mode_blocks."),
            depth_single_blocks=count("single_blocks."),
            fused_qkv="single_blocks.0.qkv.weight" in keys,
            flow_map="ref_time_in.in_layer.weight" in keys,
        )
