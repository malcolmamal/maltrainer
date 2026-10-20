"""FLUX.3 DiT building blocks; the module names match ComfyUI's (see ``model``)."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.flux3.rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.out_layer(self.silu(self.in_layer(x))))


class Modulation(nn.Module):
    """Single adaLN: (shift, scale, gate), each [B, 1, D]."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.lin(F.silu(vec[:, None, :])).chunk(3, dim=-1)
        return shift, scale, gate


class Block(nn.Module):
    """Parallel attention + SwiGLU MLP under one adaLN and one gate."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, fused_qkv: bool):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        mlp_hidden = int(hidden_size * mlp_ratio)
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, pe: torch.Tensor
    ) -> torch.Tensor:
        """``shift``/``scale``/``gate`` broadcast over x: [B, 1, D] for one mode, [B, N, D] per row."""
        B, N, _ = x.shape
        x_mod = self.pre_norm(x) * (1 + scale) + shift
        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, -1).transpose(1, 2)

        # QK-norm and RoPE in fp32, rounded once: the reference's rmsnorm_rope and ComfyUI's fused rms_rope
        q = apply_rope(self.norm.query_norm(heads(q).float()), pe).to(v.dtype)
        k = apply_rope(self.norm.key_norm(heads(k).float()), pe).to(v.dtype)
        attn = F.scaled_dot_product_attention(q, k, heads(v))
        out = self.attn_out(attn.transpose(1, 2).reshape(B, N, -1))
        gate_mlp, up = mlp.chunk(2, dim=-1)
        out = out + self.mlp_out(F.silu(gate_mlp) * up)
        return cast(torch.Tensor, x + out * gate)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec[:, None, :]).chunk(2, dim=-1)
        return cast(torch.Tensor, self.linear(self.norm_final(x) * (1 + scale) + shift))
