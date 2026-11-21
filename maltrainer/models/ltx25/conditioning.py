"""What the blocks are given besides the tokens: each stream's timestep modulations, contexts and rotary positions."""

from __future__ import annotations

from typing import Any, cast

import torch

from maltrainer.models.ltx25.blocks import Stream
from maltrainer.models.ltx25.config import LTX25Config
from maltrainer.models.ltx25.layers import AdaLayerNormSingle
from maltrainer.models.ltx25.rope import Rope, audio_positions, make_rope, video_positions


def modulation(
    adaln: AdaLayerNormSingle, sigma: torch.Tensor, prompt: AdaLayerNormSingle | None, scale: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """(timestep modulation, embedding, prompt modulation, scaled sigma) of one stream; sigma is [B] or [B, T]."""
    batch = sigma.shape[0]
    scaled = sigma.float() * scale
    modulated, embedded = adaln(scaled.flatten(), dtype)
    prompt_timestep = None
    if prompt is not None:
        peak = scaled.max(dim=1, keepdim=True).values if scaled.dim() > 1 else scaled
        prompt_timestep = prompt(peak.flatten(), dtype)[0].view(batch, 1, -1)
    return (
        modulated.view(batch, -1, modulated.shape[-1]),
        embedded.view(batch, -1, embedded.shape[-1]),
        prompt_timestep,
        scaled,
    )


def cross_modulation(adaln: AdaLayerNormSingle, values: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
    out: torch.Tensor = adaln(values.flatten(), dtype)[0]
    return out.view(batch, -1, out.shape[-1])


def build_streams(
    model: Any,  # the LTX25Transformer (typed loosely: model.py imports this module)
    sigma: torch.Tensor,
    audio_sigma: torch.Tensor | None,
    contexts: tuple[torch.Tensor, torch.Tensor | None],
    pes: dict[str, Rope],
) -> tuple[Stream, Stream | None, torch.Tensor, torch.Tensor | None]:
    """The video stream, the audio stream (None without audio) and each one's embedded timestep for the output layer."""
    config, dtype, batch = model.config, model.compute_dtype, sigma.shape[0]
    v_ts, v_embedded, v_prompt, v_scaled = modulation(
        model.adaln_single, sigma, getattr(model, "prompt_adaln_single", None), config.timestep_scale, dtype
    )
    if audio_sigma is None:
        return Stream(contexts[0], v_ts, pes["v"], v_prompt, pes["v"], v_ts, v_ts), None, v_embedded, None
    a_ts, a_embedded, a_prompt, a_scaled = modulation(
        model.audio_adaln_single,
        audio_sigma,
        getattr(model, "audio_prompt_adaln_single", None),
        config.timestep_scale,
        dtype,
    )
    factor = config.av_ca_timestep_scale / config.timestep_scale
    v_tokens, a_tokens = v_scaled.reshape(batch, -1).shape[1], a_scaled.reshape(batch, -1).shape[1]
    peak_a = (a_scaled.reshape(batch, -1).max(dim=1).values[:, None] * factor).expand(batch, v_tokens)
    peak_v = (v_scaled.reshape(batch, -1).max(dim=1).values[:, None] * factor).expand(batch, a_tokens)
    video = Stream(
        contexts[0],
        v_ts,
        pes["v"],
        v_prompt,
        pes["v_cross"],
        cross_modulation(model.av_ca_video_scale_shift_adaln_single, v_scaled, batch, dtype),
        cross_modulation(model.av_ca_a2v_gate_adaln_single, peak_a, batch, dtype),
    )
    audio = Stream(
        cast(torch.Tensor, contexts[1]),
        a_ts,
        pes["a"],
        a_prompt,
        pes["a_cross"],
        cross_modulation(model.av_ca_audio_scale_shift_adaln_single, a_scaled, batch, dtype),
        cross_modulation(model.av_ca_v2a_gate_adaln_single, peak_v, batch, dtype),
    )
    return video, audio, v_embedded, a_embedded


def positions(
    config: LTX25Config,
    frames: int,
    height: int,
    width: int,
    audio_frames: int | None,
    fps: float,
    device: torch.device,
) -> dict[str, Rope]:
    """The rotary embeddings of the video tokens and, with audio, of the audio tokens and of the cross-attention."""
    grid = video_positions(
        frames, height, width, config.vae_scale_factors, config.causal_temporal_positioning, fps, device
    )
    flags = (config.theta, config.split_rope, config.double_precision_rope)
    pes = {
        "v": make_rope(grid, config.inner_dim, config.num_heads, config.max_pos, *flags, config.use_middle_indices_grid)
    }
    if audio_frames is None:
        return pes
    audio_grid = audio_positions(audio_frames, device)
    peak = (max(config.max_pos[0], config.audio_max_pos[0]),)
    dim, heads = config.audio_cross_attention_dim, config.audio_num_heads
    pes["a"] = make_rope(
        audio_grid, config.audio_inner_dim, heads, config.audio_max_pos, *flags, config.use_middle_indices_grid
    )
    pes["v_cross"] = make_rope(grid[:, 0:1], dim, heads, peak, *flags, True)
    pes["a_cross"] = make_rope(audio_grid[:, 0:1], dim, heads, peak, *flags, True)
    return pes
