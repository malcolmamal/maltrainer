"""LTX-2.5 audio-video transformer block: per-modality self-attention, text cross-attention and feed-forward, with
audio <-> video cross-attention between them. A block without audio (``ax`` is None) is a plain video block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from maltrainer.models.ltx25.config import LTX25Config
from maltrainer.models.ltx25.layers import CrossAttention, FeedForward, ada_values, plain_rms
from maltrainer.models.ltx25.rope import Rope


@dataclass
class Stream:
    """What one modality needs in every block."""

    context: torch.Tensor
    timestep: torch.Tensor  # [B, T or 1, ada_params * dim]
    pe: Rope
    prompt_timestep: torch.Tensor | None
    cross_pe: Rope  # RoPE of the audio-video cross-attention (time only)
    cross_shift_timestep: torch.Tensor  # [B, T or 1, 4 * dim]
    cross_gate_timestep: torch.Tensor  # [B, T or 1, dim]


def modulate(x: torch.Tensor, table: torch.Tensor, timestep: torch.Tensor, rows: slice) -> torch.Tensor:
    shift, scale = ada_values(table, timestep, rows)
    return plain_rms(x) * (1 + scale) + shift


class BasicAVTransformerBlock(nn.Module):
    def __init__(self, config: LTX25Config, with_audio: bool = True):
        super().__init__()
        v, a = config.inner_dim, config.audio_inner_dim
        self.attn1 = CrossAttention(v, None, config.num_heads, config.head_dim, config.gated_attention)
        self.attn2 = CrossAttention(
            v, config.cross_attention_dim, config.num_heads, config.head_dim, config.gated_attention
        )
        self.ff = FeedForward(v, config.ff_bias)
        self.scale_shift_table = nn.Parameter(torch.zeros(config.ada_params, v))
        self.cross_attention_adaln = config.cross_attention_adaln
        if config.cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.zeros(2, v))
        self.with_audio = with_audio
        if not with_audio:
            return
        heads, head_dim = config.audio_num_heads, config.audio_head_dim
        self.audio_attn1 = CrossAttention(a, None, heads, head_dim, config.gated_attention)
        self.audio_attn2 = CrossAttention(a, config.audio_cross_attention_dim, heads, head_dim, config.gated_attention)
        self.audio_to_video_attn = CrossAttention(v, a, heads, head_dim, config.gated_attention)
        self.video_to_audio_attn = CrossAttention(a, v, heads, head_dim, config.gated_attention)
        self.audio_ff = FeedForward(a, config.audio_ff_bias)
        self.audio_scale_shift_table = nn.Parameter(torch.zeros(config.ada_params, a))
        if config.cross_attention_adaln:
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.zeros(2, a))
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.zeros(5, a))
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.zeros(5, v))

    def _text(
        self,
        x: torch.Tensor,
        attn: CrossAttention,
        table: torch.Tensor,
        prompt_table: torch.Tensor | None,
        stream: Stream,
    ) -> torch.Tensor:
        """Text cross-attention; with AdaLN in the model the query and the text are modulated by the timestep."""
        if prompt_table is None or stream.prompt_timestep is None:
            return cast(torch.Tensor, attn(plain_rms(x), stream.context))
        shift_q, scale_q, gate = ada_values(table, stream.timestep, slice(6, 9))
        shift_kv, scale_kv = (
            prompt_table[None, None].to(x.dtype) + stream.prompt_timestep.reshape(x.shape[0], -1, 2, x.shape[-1])
        ).unbind(dim=2)
        query = plain_rms(x) * (1 + scale_q) + shift_q
        return cast(torch.Tensor, attn(query, stream.context * (1 + scale_kv) + shift_kv) * gate)

    def _self_and_text(
        self,
        x: torch.Tensor,
        attn1: CrossAttention,
        attn2: CrossAttention,
        table: torch.Tensor,
        prompt_table: torch.Tensor | None,
        stream: Stream,
    ) -> torch.Tensor:
        gate_msa = ada_values(table, stream.timestep, slice(2, 3))[0]
        x = x + attn1(modulate(x, table, stream.timestep, slice(0, 2)), pe=stream.pe) * gate_msa
        return x + self._text(x, attn2, table, prompt_table, stream)

    def _feed_forward(
        self, x: torch.Tensor, ff: FeedForward, table: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            x + ff(modulate(x, table, timestep, slice(3, 5))) * ada_values(table, timestep, slice(5, 6))[0],
        )

    def _cross_modal(
        self, vx: torch.Tensor, ax: torch.Tensor, video: Stream, audio: Stream
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Audio -> video, then video -> audio attention, each modulated by both modalities' cross timesteps."""
        a_table, v_table = self.scale_shift_table_a2v_ca_audio, self.scale_shift_table_a2v_ca_video
        a_scale_a2v, a_shift_a2v, a_scale_v2a, a_shift_v2a = ada_values(a_table[:4], audio.cross_shift_timestep)
        v_scale_a2v, v_shift_a2v, v_scale_v2a, v_shift_v2a = ada_values(v_table[:4], video.cross_shift_timestep)
        ax_norm = plain_rms(ax)
        a2v = self.audio_to_video_attn(
            plain_rms(vx) * (1 + v_scale_a2v) + v_shift_a2v,
            ax_norm * (1 + a_scale_a2v) + a_shift_a2v,
            pe=video.cross_pe,
            k_pe=audio.cross_pe,
        )
        vx = vx + a2v * ada_values(v_table[4:], video.cross_gate_timestep)[0]
        v2a = self.video_to_audio_attn(
            ax_norm * (1 + a_scale_v2a) + a_shift_v2a,
            plain_rms(vx) * (1 + v_scale_v2a) + v_shift_v2a,
            pe=audio.cross_pe,
            k_pe=video.cross_pe,
        )
        return vx, ax + v2a * ada_values(a_table[4:], audio.cross_gate_timestep)[0]

    def forward(
        self, vx: torch.Tensor, ax: torch.Tensor | None, video: Stream, audio: Stream | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt = getattr(self, "prompt_scale_shift_table", None)
        vx = self._self_and_text(vx, self.attn1, self.attn2, self.scale_shift_table, prompt, video)
        if ax is not None and audio is not None:
            ax = self._self_and_text(
                ax,
                self.audio_attn1,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio,
            )
            vx, ax = self._cross_modal(vx, ax, video, audio)
            ax = self._feed_forward(ax, self.audio_ff, self.audio_scale_shift_table, audio.timestep)
        return self._feed_forward(vx, self.ff, self.scale_shift_table, video.timestep), ax
