"""The MiniMax H3 Image transformer: H3's blocks with no audio stream and a bank of per-level output heads.

The checkpoint's ``final_layer.video_out`` is ``[levels * 96, hidden]``: row block 0 is a full head and blocks 1..n-1
are offsets from it, so the head for noise level ``k`` (the step from grid sigma k to k + 1) is ``rows[0] + rows[k]``
(``rows[0]`` alone for level 0). That is what ComfyUI's ``_pdd_head`` computes for a one-level span.
"""

from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.minimax_h3.model import MiniMaxH3AdaLNProjection, MiniMaxH3Backbone
from maltrainer.models.minimax_h3.model_config import MiniMaxH3TransformerConfig


class MiniMaxH3ImageFinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, levels: int):
        super().__init__()
        self.levels = levels
        self.patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim, 2 * config.hidden_size, apply_silu=config.adaln_t_table_size is None
        )
        self.video_out = nn.Linear(config.hidden_size, levels * self.patch_dim)

    def head(self, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (weight, bias) that predicts the step leaving noise level ``level``."""
        if not 0 <= level < self.levels:
            raise ValueError(f"noise level {level} is outside the checkpoint's {self.levels} output heads")
        weight = self.video_out.weight.view(self.levels, self.patch_dim, -1)
        bias = self.video_out.bias.view(self.levels, self.patch_dim)
        if level == 0:
            return weight[0], bias[0]
        return weight[0] + weight[level], bias[0] + bias[level]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        timing: tuple[torch.Tensor, torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        timestep_indices, video_indices = timing
        shift, scale = self.adaln_proj(timestep_embedding).chunk(2, dim=-1)
        shift, scale = shift.to(hidden_states.dtype), scale.to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states) * (1.0 + scale.index_select(0, timestep_indices))
        hidden_states = hidden_states + shift.index_select(0, timestep_indices)
        weight, bias = self.head(level)
        hidden_states = hidden_states.index_select(1, video_indices).to(weight.dtype)
        return torch.nn.functional.linear(hidden_states, weight, bias)


class MiniMaxH3ImageTransformer(MiniMaxH3Backbone):
    """[text | image] packed sequence in, the image tokens' ``x0 - noise`` prediction out."""

    label = "MiniMax H3 Image"

    def __init__(self, config: MiniMaxH3TransformerConfig | None = None, levels: int = 6):
        super().__init__(config)
        self.final_layer = MiniMaxH3ImageFinalLayer(self.config, levels)

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        text_indices: torch.Tensor,
        level: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rotary_emb = self.rope(position_ids)
        video = self.video_patch_proj(video_hidden_states.to(self.video_patch_proj.weight.dtype))
        text = self.condition_proj(encoder_hidden_states.to(self.condition_proj.weight.dtype))
        text = self.token_refiner(text, self.gradient_checkpointing)
        hidden_states = text.new_zeros((text.shape[0], position_ids.shape[0], text.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text)
        hidden_states = hidden_states.index_copy(1, video_indices, video.to(text.dtype))
        hidden_states, timestep_embedding = self.run_blocks(
            hidden_states, (timestep, timestep_indices, token_tags), rotary_emb, attention_mask
        )
        out: torch.Tensor = self.final_layer(
            hidden_states, timestep_embedding, (timestep_indices, video_indices), level
        )
        return out
