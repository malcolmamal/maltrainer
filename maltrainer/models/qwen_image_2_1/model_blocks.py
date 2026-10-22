"""Attention and the transformer block of Qwen-Image-2.1."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.qwen_image_2_1.model_layers import RMSNorm, SwiGLUFeedForward, apply_rotary_emb_qwen


class Attention(nn.Module):
    """Multi-head attention module with rotary embeddings and QK-norm."""

    def __init__(self, dim: int = 4096, heads: int = 32, dim_head: int = 128, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(b, s, self.heads, self.dim_head)
        k = self.to_k(hidden_states).view(b, s, self.heads, self.dim_head)
        v = self.to_v(hidden_states).view(b, s, self.heads, self.dim_head)
        q = self.norm_q(q)
        k = self.norm_k(k)

        if rotary_emb is not None:
            q = apply_rotary_emb_qwen(q, rotary_emb)
            k = apply_rotary_emb_qwen(k, rotary_emb)

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, -1)
        out = self.to_out[0](out)
        return cast(torch.Tensor, out)


class TransformerBlock(nn.Module):
    """Single-stream Qwen 2.1 Transformer block."""

    def __init__(self, dim: int = 4096, heads: int = 32, dim_head: int = 128, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        if target_token_mask is None:
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        else:
            real_s, zero_s = scale[:-1].unsqueeze(1), scale[-1:].unsqueeze(0)
            scale = torch.where(target_token_mask.view(1, -1, 1), real_s, zero_s)
            real_g, zero_g = gate[:-1].unsqueeze(1), gate[-1:].unsqueeze(0)
            gate = torch.where(target_token_mask.view(1, -1, 1), real_g, zero_g)
        return hidden_states * (1.0 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_out = self.attn(img_modulated, rotary_emb=rotary_emb, attn_mask=attn_mask)
        hidden_states = hidden_states + img_gate1.tanh() * attn_out

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)
        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm Continuous module."""

    def __init__(self, embedding_dim: int = 4096, conditioning_embedding_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        if target_token_mask is None:
            scale = scale.unsqueeze(1)
        else:
            real_s, zero_s = scale[:-1].unsqueeze(1), scale[-1:].unsqueeze(0)
            scale = torch.where(target_token_mask.view(1, -1, 1), real_s, zero_s)
        return cast(torch.Tensor, self.norm(hidden_states) * (1.0 + scale))
