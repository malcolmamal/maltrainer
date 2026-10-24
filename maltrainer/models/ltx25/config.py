"""
LTX-2.5 DiT hyper-parameters.

Structural sizes are read off the tensors of ComfyUI's single file, as ComfyUI does. The rest (RoPE flavour,
timestep scales, causal positions) lives in the file's ``config`` metadata under ``transformer``, whose key names
are ComfyUI's constructor arguments. Without metadata the LTX-2.3 values are used, which LTX-2.5 shares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from typing import Any

AUDIO_CHANNELS = 8  # audio latent channels
AUDIO_BINS = 16  # latent mel bins per channel
VIDEO_HEAD_DIM, AUDIO_HEAD_DIM = 128, 64


@dataclass(frozen=True)
class ConnectorConfig:
    """One modality's 1D text connector."""

    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 8
    registers: int = 128
    gated: bool = True
    ff_bias: bool = True

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class LTX25Config:
    in_channels: int = 128
    audio_in_channels: int = 128
    num_layers: int = 48
    num_heads: int = 32
    head_dim: int = 128
    audio_num_heads: int = 32
    audio_head_dim: int = 64
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048
    theta: float = 10000.0
    max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_max_pos: tuple[int, ...] = (20,)
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    causal_temporal_positioning: bool = True
    use_middle_indices_grid: bool = True
    split_rope: bool = True
    double_precision_rope: bool = True
    timestep_scale: float = 1000.0
    av_ca_timestep_scale: float = 1000.0
    gated_attention: bool = True
    cross_attention_adaln: bool = True
    prompt_adaln: bool = True
    caption_projection: bool = False
    ff_bias: bool = True
    audio_ff_bias: bool = True
    video_connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    audio_connector: ConnectorConfig = field(default_factory=lambda: ConnectorConfig(head_dim=AUDIO_HEAD_DIM))

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_heads * self.audio_head_dim

    @property
    def ada_params(self) -> int:
        return 9 if self.cross_attention_adaln else 6


def transformer_metadata(metadata: dict[str, str] | None) -> dict[str, Any]:
    """The ``transformer`` section of the file's ``config`` metadata, or nothing."""
    if not metadata or "config" not in metadata:
        return {}
    section = json.loads(metadata["config"]).get("transformer", {})
    return section if isinstance(section, dict) else {}


def _count(state: dict[str, Any], prefix: str, position: int) -> int:
    indices = [int(key.split(".")[position]) for key in state if key.startswith(prefix)]
    return 1 + max(indices) if indices else 0


def _connector(state: dict[str, Any], name: str, head_dim: int, default: ConnectorConfig) -> ConnectorConfig:
    registers = state.get(f"{name}.learnable_registers")
    if registers is None:
        return default
    prefix = f"{name}.transformer_1d_blocks."
    return ConnectorConfig(
        num_heads=registers.shape[1] // head_dim,
        head_dim=head_dim,
        num_layers=_count(state, prefix, 2),
        registers=registers.shape[0],
        gated=f"{prefix}0.attn1.to_gate_logits.weight" in state,
        ff_bias=f"{prefix}0.ff.net.0.proj.bias" in state,
    )


def _overrides(meta: dict[str, Any]) -> dict[str, Any]:
    """Metadata keys (ComfyUI's constructor names) as config fields."""
    names = {
        "positional_embedding_theta": "theta",
        "positional_embedding_max_pos": "max_pos",
        "audio_positional_embedding_max_pos": "audio_max_pos",
        "causal_temporal_positioning": "causal_temporal_positioning",
        "use_middle_indices_grid": "use_middle_indices_grid",
        "timestep_scale_multiplier": "timestep_scale",
        "av_ca_timestep_scale_multiplier": "av_ca_timestep_scale",
        "vae_scale_factors": "vae_scale_factors",
    }
    out = {field_name: meta[key] for key, field_name in names.items() if key in meta}
    for key in ("max_pos", "audio_max_pos", "vae_scale_factors"):
        if key in out:
            out[key] = tuple(out[key])
    if "rope_type" in meta:
        out["split_rope"] = meta["rope_type"] == "split"
    if "frequencies_precision" in meta:
        out["double_precision_rope"] = meta["frequencies_precision"] == "float64"
    return out


def config_from_state(state: dict[str, Any], metadata: dict[str, str] | None = None) -> LTX25Config:
    """The config of a ComfyUI LTX-2.x audio-video DiT file (tensors for sizes, metadata for the rest)."""
    meta = transformer_metadata(metadata)
    inner = state["patchify_proj.weight"].shape[0]
    audio_inner = state["audio_patchify_proj.weight"].shape[0] if "audio_patchify_proj.weight" in state else 2048
    heads, audio_heads = meta.get("num_attention_heads", 32), meta.get("audio_num_attention_heads", 32)
    blocks = "transformer_blocks.0."
    base = LTX25Config()
    config = replace(
        base,
        in_channels=state["patchify_proj.weight"].shape[1],
        num_layers=_count(state, "transformer_blocks.", 1),
        num_heads=heads,
        head_dim=meta.get("attention_head_dim", inner // heads),
        audio_num_heads=audio_heads,
        audio_head_dim=meta.get("audio_attention_head_dim", audio_inner // audio_heads),
        cross_attention_dim=state[f"{blocks}attn2.to_k.weight"].shape[1],
        gated_attention=f"{blocks}attn1.to_gate_logits.weight" in state,
        cross_attention_adaln=f"{blocks}prompt_scale_shift_table" in state,
        prompt_adaln="prompt_adaln_single.linear.weight" in state,
        caption_projection="caption_projection.linear_1.weight" in state,
        ff_bias=f"{blocks}ff.net.0.proj.bias" in state,
        audio_ff_bias=f"{blocks}audio_ff.net.0.proj.bias" in state,
        video_connector=_connector(
            state,
            "video_embeddings_connector",
            meta.get("connector_attention_head_dim", VIDEO_HEAD_DIM),
            base.video_connector,
        ),
        audio_connector=_connector(
            state,
            "audio_embeddings_connector",
            meta.get("audio_connector_attention_head_dim", AUDIO_HEAD_DIM),
            base.audio_connector,
        ),
    )
    if f"{blocks}audio_attn2.to_k.weight" in state:
        config = replace(config, audio_cross_attention_dim=state[f"{blocks}audio_attn2.to_k.weight"].shape[1])
    known = {f.name for f in fields(LTX25Config)}
    return replace(config, **{k: v for k, v in _overrides(meta).items() if k in known})
