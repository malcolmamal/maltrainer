"""The projection from Gemma's hidden states to LTX-2.5 text features (it ships inside the text-encoder file)."""

from __future__ import annotations

import math

import torch
from torch import nn

NORM_EPS = 1e-6


class DualProjection(nn.Module):
    """All hidden states -> per-layer RMS norm -> one linear per modality (comfy's DualLinearProjection)."""

    def __init__(self, in_dim: int, video_dim: int, audio_dim: int, video_bias: bool = True, audio_bias: bool = True):
        super().__init__()
        self.audio_aggregate_embed = nn.Linear(in_dim, audio_dim, bias=audio_bias)
        self.video_aggregate_embed = nn.Linear(in_dim, video_dim, bias=video_bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states [B, T, hidden, layers + 1] -> [B, T, video_dim + audio_dim]."""
        hidden = states.shape[2]
        dtype = self.video_aggregate_embed.weight.dtype
        normalised = states.float() * torch.rsqrt(states.float().pow(2).mean(dim=2, keepdim=True) + NORM_EPS)
        flat = normalised.flatten(2).to(dtype if dtype.is_floating_point else states.dtype)
        video = self.video_aggregate_embed(flat * math.sqrt(self.video_aggregate_embed.out_features / hidden))
        audio = self.audio_aggregate_embed(flat * math.sqrt(self.audio_aggregate_embed.out_features / hidden))
        return torch.cat((video, audio), dim=-1)


def stack_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The model's (layers + 1) hidden states [B, T, hidden] -> [B, T, hidden, layers + 1]."""
    return torch.stack(hidden_states, dim=-1)
