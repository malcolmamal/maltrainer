"""Token refiner block, transformer block and final layer of the MiniMax H3 transformer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from maltrainer.models.minimax_h3.model_config import (
    MINIMAX_H3_MODALITY_COUNT,
    MiniMaxH3TransformerConfig,
    MiniMaxH3TransformerOutput,
)
from maltrainer.models.minimax_h3.model_layers import MiniMaxH3AdaLNProjection, MiniMaxH3Attention, MiniMaxH3FeedForward


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return cast(torch.Tensor, hidden_states + self.mlp(self.norm2(hidden_states)))


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            6 * config.hidden_size * MINIMAX_H3_MODALITY_COUNT,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation = self.adaln_proj(timestep_embedding).view(-1, 6 * self.hidden_size)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        hidden_dtype = hidden_states.dtype
        shift_attn = shift_attn.to(hidden_dtype)
        scale_attn = scale_attn.to(hidden_dtype)
        gate_attn = gate_attn.to(hidden_dtype)
        shift_mlp = shift_mlp.to(hidden_dtype)
        scale_mlp = scale_mlp.to(hidden_dtype)
        gate_mlp = gate_mlp.to(hidden_dtype)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_attn.index_select(0, adaln_indices))
        norm_hidden_states = norm_hidden_states + shift_attn.index_select(0, adaln_indices)
        attention = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = hidden_states + gate_attn.index_select(0, adaln_indices) * attention

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp.index_select(0, adaln_indices))
        norm_hidden_states = norm_hidden_states + shift_mlp.index_select(0, adaln_indices)
        feed_forward = self.mlp(norm_hidden_states)
        return cast(torch.Tensor, hidden_states + gate_mlp.index_select(0, adaln_indices) * feed_forward)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            2 * config.hidden_size,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.video_out = nn.Linear(config.hidden_size, video_patch_dim)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
    ) -> MiniMaxH3TransformerOutput:
        shift, scale = self.adaln_proj(timestep_embedding).chunk(2, dim=-1)
        shift = shift.to(hidden_states.dtype)
        scale = scale.to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1.0 + scale.index_select(0, timestep_indices))
        hidden_states = hidden_states + shift.index_select(0, timestep_indices)
        video_dtype = self.video_out.weight.dtype
        audio_dtype = self.audio_out.weight.dtype
        video = self.video_out(hidden_states.to(video_dtype)).index_select(1, video_indices)
        audio = self.audio_out(hidden_states.to(audio_dtype)).index_select(1, audio_indices)
        return MiniMaxH3TransformerOutput(video=video, audio=audio)
