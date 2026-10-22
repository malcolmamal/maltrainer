"""Startup summary of the H3 text tower and the checkpoint-to-model key map of the Qwen3-VL text encoder."""

from __future__ import annotations

from typing import Any


def describe_text_config(text_config: Any) -> str:
    """Compact summary of the resolved text tower, for a startup sanity check.

    Read defensively: transformers has reshuffled where RoPE settings live between
    versions (rope_theta/rope_scaling vs. a merged rope_parameters dict), and a log line
    must never be the thing that fails a training run.
    """
    fields = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim"]
    parts = [f"{name}={getattr(text_config, name, '?')}" for name in fields]
    rope = getattr(text_config, "rope_parameters", None)
    if rope is None:
        theta = getattr(text_config, "rope_theta", "?")
        rope = {"rope_theta": theta, "rope_scaling": getattr(text_config, "rope_scaling", None)}
    parts.append(f"rope={rope}")
    return " ".join(parts)


def text_encoder_key(source_prefix: str) -> str:
    if source_prefix.startswith("visual"):
        return source_prefix
    if source_prefix == "model":
        return "language_model"
    if source_prefix.startswith("model."):
        return f"language_model.{source_prefix.removeprefix('model.')}"
    return source_prefix
