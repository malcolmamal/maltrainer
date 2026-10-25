"""Fibo 2.0 transformer blocks: Z-Image's block (with an injection variant) and the Perceiver."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from maltrainer.models.fibo2.config import ADALN_DIM, Fibo2Config
from maltrainer.models.fibo2.layers import Attention, FeedForward, JointAttention, RMSNorm

Span = tuple[int, int]
Values = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # scale_msa, gate_msa, scale_mlp, gate_mlp


def modulate(x: torch.Tensor, noisy: torch.Tensor, clean: torch.Tensor | None, span: Span | None) -> torch.Tensor:
    """x * noisy on the tokens of ``span`` (the image) and x * clean on the rest; x * noisy on all with no span."""
    if span is None or clean is None:
        return x * noisy[:, None]
    start, end = span
    return torch.cat(
        (x[:, :start] * clean[:, None], x[:, start:end] * noisy[:, None], x[:, end:] * clean[:, None]), dim=1
    )


def modulation_values(mod: torch.Tensor) -> Values:
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    return 1 + scale_msa, gate_msa.tanh(), 1 + scale_mlp, gate_mlp.tanh()


class Block(nn.Module):
    """Attention and SwiGLU under tanh-gated adaLN. An injection block also cross-attends to its own text bundle."""

    def __init__(self, config: Fibo2Config, injection: bool):
        super().__init__()
        dim, eps = config.dim, config.norm_eps
        self.attention = JointAttention(dim, config.n_heads, eps)
        self.feed_forward = FeedForward(dim, config.ffn_dim)
        self.attention_norm1, self.attention_norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.ffn_norm1, self.ffn_norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(ADALN_DIM, 4 * dim))
        self.cross_attention: Attention | None = None
        if injection:
            self.cross_attn_norm1, self.cross_attn_norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
            self.cross_attention = Attention(dim, config.n_heads, config.cap_feat_dim, eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        adaln: torch.Tensor,
        span: Span | None,
        text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """adaln: [1, 256] for every token, or [2, 256] = [noisy; clean] when ``span`` marks the noisy tokens."""
        mod = self.adaLN_modulation(adaln)
        if span is None:
            noisy, clean = modulation_values(mod), None
        else:
            noisy, clean = (modulation_values(part) for part in mod.chunk(2, dim=0))

        def gate(values: int, y: torch.Tensor) -> torch.Tensor:
            return modulate(y, noisy[values], None if clean is None else clean[values], span)

        x = x + gate(1, self.attention_norm2(self.attention(gate(0, self.attention_norm1(x)), *rope)))
        if self.cross_attention is not None and text is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), text)
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(y)
        return x + gate(3, self.ffn_norm2(self.feed_forward(gate(2, self.ffn_norm1(x)))))


class PerceiverLayer(nn.Module):
    def __init__(self, config: Fibo2Config):
        super().__init__()
        dim, eps = config.dim, config.norm_eps
        self.cross_attn_norm = RMSNorm(dim, eps)
        self.cross_attention = Attention(dim, config.n_heads, config.cap_feat_dim, eps)
        self.self_attn_norm = RMSNorm(dim, eps)
        self.self_attention = Attention(dim, config.n_heads, dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, config.ffn_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attention(self.cross_attn_norm(x), context)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y)
        return cast(torch.Tensor, x + self.ffn(self.ffn_norm(x)))


class Perceiver(nn.Module):
    """Squeezes the first text bundle into gist tokens, the head of the DiT's sequence."""

    def __init__(self, config: Fibo2Config):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(config.max_gist_tokens, config.dim))
        self.layers = nn.ModuleList([PerceiverLayer(config) for _ in range(config.perceiver_layers)])
        self.norm_out = RMSNorm(config.dim, config.norm_eps)
