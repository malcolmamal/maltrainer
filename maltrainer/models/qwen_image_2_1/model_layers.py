"""Embedding, normalisation and feed-forward layers of the Qwen-Image-2.1 transformer."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class TemporalTimesteps(nn.Module):
    """Sinusoidal timestep embedding generator."""

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.get_buffer("freqs").to(timestep.device)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(timestep.dtype)


class TimestepEmbedder(nn.Module):
    """Timestep MLP projection."""

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 4096):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear_2(self.act(self.linear_1(x))))


class TimestepProjEmbeddings(nn.Module):
    """Combined Timestep Projection and Embedding block."""

    def __init__(self, embedding_dim: int = 4096):
        super().__init__()
        self.time_proj = TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedder(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj(timestep)
        return cast(torch.Tensor, self.timestep_embedder(proj.to(dtype=hidden_states.dtype)))


class ZeroCenterRMSNorm(nn.Module):
    """Zero-centered RMSNorm for text projection input."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1.0)).to(input_dtype)


class TextProjection(nn.Module):
    """Text embedding projection layer."""

    def __init__(self, context_in_dim: int = 4096, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return cast(torch.Tensor, self.out_layer(hidden_states))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU MLP layer."""

    def __init__(self, hidden_size: int = 4096, mlp_hidden_size: int = 12288):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, mlp_hidden_size * 2, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return cast(torch.Tensor, self.out(F.silu(gate) * up))


class RMSNorm(nn.Module):
    """RMSNorm for attention queries and keys."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=-1, eps=self.eps).to(x.dtype)
        return cast(torch.Tensor, norm * (self.weight.shape[-1] ** 0.5) * self.weight)


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding."""
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # [B, 1, S, D/2]
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)
