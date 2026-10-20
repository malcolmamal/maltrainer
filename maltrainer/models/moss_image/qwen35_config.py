"""
Qwen3.5 text model sizes, read off the weights (see ``text_encoder``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Qwen35TextConfig:
    """Qwen3.5-9B's text model (comfy/text_encoders/qwen35.py QWEN35_MODELS["qwen35_9b"])."""

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    full_attention_layers: tuple[int, ...] = tuple(i for i in range(32) if (i + 1) % 4 == 0)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> Qwen35TextConfig:
        """Read the sizes off the weights, so a mis-picked file fails here and not deep in a matmul."""
        if "layers.0.linear_attn.A_log" not in state:
            raise ValueError("Not a Qwen3.5 text encoder (no layers.0.linear_attn.A_log)")
        layers = sorted({int(k.split(".", 2)[1]) for k in state if k.startswith("layers.")})
        full = tuple(i for i in layers if f"layers.{i}.self_attn.k_proj.weight" in state)
        first_full = full[0] if full else 3
        head_dim = state[f"layers.{first_full}.self_attn.q_norm.weight"].shape[0]
        value_heads = state["layers.0.linear_attn.A_log"].shape[0]
        value_dim = state["layers.0.linear_attn.in_proj_z.weight"].shape[0]
        conv_dim = state["layers.0.linear_attn.conv1d.weight"].shape[0]
        key_head_dim = cls.linear_key_head_dim
        return cls(
            vocab_size=state["embed_tokens.weight"].shape[0],
            hidden_size=state["embed_tokens.weight"].shape[1],
            intermediate_size=state["layers.0.mlp.gate_proj.weight"].shape[0],
            num_hidden_layers=len(layers),
            num_attention_heads=state[f"layers.{first_full}.self_attn.q_proj.weight"].shape[0] // (2 * head_dim),
            num_key_value_heads=state[f"layers.{first_full}.self_attn.k_proj.weight"].shape[0] // head_dim,
            head_dim=head_dim,
            linear_num_key_heads=(conv_dim - value_dim) // (2 * key_head_dim),
            linear_num_value_heads=value_heads,
            linear_value_head_dim=value_dim // value_heads,
            conv_kernel_size=state["layers.0.linear_attn.conv1d.weight"].shape[-1],
            full_attention_layers=full,
        )
