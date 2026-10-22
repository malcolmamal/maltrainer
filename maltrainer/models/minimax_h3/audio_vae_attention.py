"""The attention projection down to the audio latent width."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return cast(torch.Tensor, self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        bias = torch.cat((self.q_bias, self.get_buffer("zero_k_bias"), self.v_bias)).to(x.dtype)
        qkv = F.linear(x, self.qkv.weight.to(x.dtype), bias)
        q, k, v = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # mean over heads, then pool the width down to the latent width
        x = F.adaptive_avg_pool1d(x.mean(dim=1), self.out_dim)
        return cast(torch.Tensor, self.proj(x))


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return cast(torch.Tensor, x + self.mlp(self.norm2(x)))
