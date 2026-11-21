"""Visage-Image's DiT blocks: adaLN-Zero, SwiGLU, QK-RMSNorm attention over [text, image], spiral RoPE on the image."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maltrainer.models.visage.config import HEAD_DIM

Rope = tuple[Tensor, Tensor]


class SinusoidalEmbedding(nn.Module):
    angular_speeds: Tensor

    def __init__(self, min_freq: float, max_freq: float, dims: int) -> None:
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), dims // 2))
        self.register_buffer("angular_speeds", 2.0 * torch.pi * freqs)

    def forward(self, t: Tensor) -> Tensor:
        """fp32 whatever the model's dtype, as in training: the fastest frequencies turn 600 rad over t in [0, 1]."""
        angles = self.angular_speeds.float() * t.float()
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def spiral_rope_tables(
    height: int, width: int, device: torch.device, base: float = 10000.0, directions: int = 4
) -> Rope:
    """cos / sin tables [1, 1, H*W, HEAD_DIM / 2]; rotary pairs cycle through ``directions`` orientations in [0, pi)."""
    pairs = HEAD_DIM // 2
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    index = torch.arange(pairs, device=device)
    freq_count = max((pairs + directions - 1) // directions, 1)
    inv_freq = 1.0 / (base ** ((index // directions).float() / max(freq_count - 1, 1)))
    theta = torch.pi * (index % directions).float() / directions
    angles = (coords @ torch.stack((theta.cos(), theta.sin()), dim=-1).T) * inv_freq
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x: Tensor, rope: Rope) -> Tensor:
    cos, sin = (r.to(x.dtype) for r in rope)
    first, second = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """SwiGLU: fc1 produces value and gate together."""

    def __init__(self, width: int, ratio: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width * ratio * 2)
        self.fc2 = nn.Linear(width * ratio, width)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return cast(Tensor, self.fc2(F.silu(value) * gate))


class Attention(nn.Module):
    """Joint attention over [text, image]; RoPE on the image tokens only."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // heads, eps=1e-6)

    def forward(self, x: Tensor, n_text: int, rope: Rope) -> Tensor:
        qkv = cast(Tensor, self.qkv(x))
        q, k, v = (t.reshape(*t.shape[:-1], self.heads, -1).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q = torch.cat((q[..., :n_text, :], apply_rope(q[..., n_text:, :], rope)), dim=-2)
        k = torch.cat((k[..., :n_text, :], apply_rope(k[..., n_text:, :], rope)), dim=-2)
        return cast(Tensor, self.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2)))


class Block(nn.Module):
    def __init__(self, width: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        modulation = nn.Linear(width, 6 * width)
        self.mod = nn.Sequential(nn.SiLU(), modulation)
        self.attn = Attention(width, width // HEAD_DIM)
        self.mlp = MLP(width, mlp_ratio)
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(self, x: Tensor, cond: Tensor, n_text: int, rope: Rope) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(cond).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1), n_text, rope)
        return cast(Tensor, x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2)))
