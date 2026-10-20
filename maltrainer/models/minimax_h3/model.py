"""
Native MiniMax H3 Transformer implementation.

The config and tags live in ``model_config``, the embedding/attention/feed-forward layers in ``model_layers`` and the
transformer blocks in ``model_blocks``. The token refiner and the transformer stay here, next to the
``checkpoint`` they call (the tests patch it on this module); every public name is re-exported.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.minimax_h3.model_blocks import (
    MiniMaxH3FinalLayer,
    MiniMaxH3TokenRefinerBlock,
    MiniMaxH3TransformerBlock,
)
from maltrainer.models.minimax_h3.model_config import (
    MINIMAX_H3_MODALITY_COUNT,
    MiniMaxH3TokenTag,
    MiniMaxH3TransformerConfig,
    MiniMaxH3TransformerOutput,
)
from maltrainer.models.minimax_h3.model_layers import (
    MiniMaxH3AdaLNProjection,
    MiniMaxH3Attention,
    MiniMaxH3FeedForward,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TimeEmbedder,
    _apply_rotary_emb,
    get_timestep_embedding,
)

__all__ = [
    "MINIMAX_H3_MODALITY_COUNT",
    "MiniMaxH3AdaLNProjection",
    "MiniMaxH3Attention",
    "MiniMaxH3FeedForward",
    "MiniMaxH3FinalLayer",
    "MiniMaxH3RotaryPosEmbed",
    "MiniMaxH3TimeEmbedder",
    "MiniMaxH3TokenRefiner",
    "MiniMaxH3TokenRefinerBlock",
    "MiniMaxH3TokenTag",
    "MiniMaxH3Transformer",
    "MiniMaxH3TransformerBlock",
    "MiniMaxH3TransformerConfig",
    "MiniMaxH3TransformerOutput",
    "_apply_rotary_emb",
    "get_timestep_embedding",
]


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(config) for _ in range(config.num_refiner_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if torch.is_grad_enabled() and gradient_checkpointing:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return cast(torch.Tensor, self.final_norm(hidden_states))


class MiniMaxH3Transformer(nn.Module):
    """Single-stream MiniMax H3 video-audio transformer."""

    def __init__(self, config: MiniMaxH3TransformerConfig | None = None):
        super().__init__()
        self.config = config or MiniMaxH3TransformerConfig()
        config = self.config
        rotary_dim = 2 * 3 * config.rope_freq_dim
        if rotary_dim > config.attention_head_dim:
            raise ValueError(f"H3 rotary width {rotary_dim} exceeds attention head width {config.attention_head_dim}")

        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.video_patch_proj = nn.Linear(video_patch_dim, config.hidden_size)
        self.audio_patch_proj = nn.Linear(config.audio_in_channels, config.hidden_size)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size)
        if config.adaln_t_table_size is None:
            self.time_embedder: MiniMaxH3TimeEmbedder | None = MiniMaxH3TimeEmbedder(
                config.freq_dim,
                config.time_embed_hidden_dim,
                config.time_embed_dim,
            )
        else:
            if config.adaln_t_table_size < 2:
                raise ValueError("H3 AdaLN timestep table must contain at least two rows")
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.zeros(config.adaln_t_table_size, config.time_embed_dim),
                persistent=True,
            )
        self.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.blocks = nn.ModuleList([MiniMaxH3TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(config)
        self.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.time_embedder is not None:
            return cast(torch.Tensor, self.time_embedder(timestep))
        table = self.get_buffer("adaln_t_table").float()
        position = timestep.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=table.shape[0] - 1)
        fraction = (position - lower.float()).unsqueeze(1)
        embedding = table.index_select(0, lower) * (1.0 - fraction) + table.index_select(0, upper) * fraction
        return embedding.to(timestep.device)

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MiniMaxH3TransformerOutput:
        sequence_length = position_ids.shape[0]
        rotary_emb = self.rope(position_ids)
        video = self.video_patch_proj(video_hidden_states.to(self.video_patch_proj.weight.dtype))
        audio = self.audio_patch_proj(audio_hidden_states.to(self.audio_patch_proj.weight.dtype))
        text = self.condition_proj(encoder_hidden_states.to(self.condition_proj.weight.dtype))
        text = self.token_refiner(text, self.gradient_checkpointing)
        hidden_states = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text)
        hidden_states = hidden_states.index_copy(1, video_indices, video.to(text.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio.to(text.dtype))

        timestep_embedding = self._time_embedding(timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_COUNT + token_tags.clamp(min=0)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    timestep_embedding,
                    adaln_indices,
                    rotary_emb,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, timestep_embedding, adaln_indices, rotary_emb, attention_mask)

        output: MiniMaxH3TransformerOutput = self.final_layer(
            hidden_states, timestep_embedding, timestep_indices, video_indices, audio_indices
        )
        return output
