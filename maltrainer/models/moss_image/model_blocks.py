"""
MOSS-Image DiT building blocks: RMSNorm, QK-normed attention, the text embedder and the joint block (see ``model``).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.moss_image.model_math import apply_rope, situ

_T = torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def attention(q: _T, k: _T, v: _T, q_norm: RMSNorm, k_norm: RMSNorm, pe: _T, heads: int) -> torch.Tensor:
    """[B, N, H*D] q/k/v -> [B, N, H*D]. QK-norm and RoPE in fp32, rounded once, as ComfyUI's
    fused rms_rope_split_half kernel does at inference."""
    B, N, _ = q.shape
    q = apply_rope(q_norm(q.view(B, N, heads, -1).float()), pe).to(v.dtype)
    k = apply_rope(k_norm(k.view(B, N, heads, -1).float()), pe).to(v.dtype)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.view(B, N, heads, -1).transpose(1, 2))
    return out.transpose(1, 2).reshape(B, N, -1)


class TextBlock(nn.Module):
    """Unmodulated pre-norm block of the text embedder; its Linears carry biases."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads))
        y = self.norm2(x)
        return cast(torch.Tensor, x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y))))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, fused_qkv: bool):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({n: nn.Linear(dim, 3 * dim if n == "qkv" else dim, bias=False) for n in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self, x: _T, shift_a: _T, scale_a: _T, gate_a: _T, shift_f: _T, scale_f: _T, gate_f: _T, pe: _T
    ) -> torch.Tensor:
        """The six adaLN vectors are [B, 1, D], shared by every token."""
        y = self.norm1(x) * (1 + scale_a) + shift_a
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        x = x + self.proj["o"](attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads)) * gate_a
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        return cast(torch.Tensor, x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f)
