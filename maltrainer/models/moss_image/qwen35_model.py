"""
Qwen3.5 text model: the decoder layer and the stack (see ``text_encoder``).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.moss_image.qwen35_config import Qwen35TextConfig
from maltrainer.models.moss_image.qwen35_delta import GatedDeltaNet
from maltrainer.models.moss_image.qwen35_layers import MLP, GatedAttention, RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35TextConfig, index: int):
        super().__init__()
        if index in config.full_attention_layers:
            self.self_attn = GatedAttention(config)
        else:
            self.linear_attn = GatedDeltaNet(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin) if hasattr(self, "self_attn") else self.linear_attn(h)
        x = x + h
        return cast(torch.Tensor, x + self.mlp(self.post_attention_layernorm(x)))


class Qwen35TextModel(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, add=True)
        self.compute_dtype = torch.float32

    def rope(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-only positions: Qwen3.5's interleaved MRoPE collapses to 1D RoPE (all three axes equal)."""
        dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.arange(length, device=device).float()[:, None] * inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed_tokens, nn.Embedding):
            x = F.embedding(input_ids, self.embed_tokens.weight).to(self.compute_dtype)
        else:
            cast(Any, self.embed_tokens).out_dtype = self.compute_dtype
            x = self.embed_tokens(input_ids)
        cos, sin = self.rope(input_ids.shape[1], input_ids.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return cast(torch.Tensor, self.norm(x))
