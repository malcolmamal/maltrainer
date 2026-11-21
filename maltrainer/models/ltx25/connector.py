"""The 1D text connector: pads the projected text with learnable registers and runs a short transformer over it."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from maltrainer.models.ltx25.config import ConnectorConfig
from maltrainer.models.ltx25.layers import CrossAttention, FeedForward, plain_rms
from maltrainer.models.ltx25.rope import make_rope

MIN_TOKENS = 1024  # the registers fill the text up to at least this many tokens
POSITION_RANGE = (4096,)


class ConnectorBlock(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.attn1 = CrossAttention(config.inner_dim, None, config.num_heads, config.head_dim, config.gated)
        self.ff = FeedForward(config.inner_dim, config.ff_bias)

    def forward(self, x: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor, bool]) -> torch.Tensor:
        x = self.attn1(plain_rms(x), pe=pe) + x
        return cast(torch.Tensor, self.ff(plain_rms(x)) + x)


class Embeddings1DConnector(nn.Module):
    def __init__(self, config: ConnectorConfig, theta: float = 10000.0, split_rope: bool = True, double: bool = True):
        super().__init__()
        self.config, self.theta, self.split_rope, self.double_precision = config, theta, split_rope, double
        self.transformer_1d_blocks = nn.ModuleList([ConnectorBlock(config) for _ in range(config.num_layers)])
        self.learnable_registers = nn.Parameter(torch.zeros(config.registers, config.inner_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, N, inner] projected text -> [B, max(1024, N) rounded up to whole register tiles, inner]."""
        tokens, registers = hidden.shape[1], self.config.registers
        tiles = math.ceil(max(MIN_TOKENS, tokens) / registers)
        tiled = self.learnable_registers.to(hidden).repeat(tiles, 1)
        hidden = torch.cat((hidden, tiled[tokens:].unsqueeze(0).expand(hidden.shape[0], -1, -1)), dim=1)
        grid = torch.arange(hidden.shape[1], dtype=torch.float32, device=hidden.device)[None, None, :]
        pe = make_rope(
            grid,
            self.config.inner_dim,
            self.config.num_heads,
            POSITION_RANGE,
            self.theta,
            self.split_rope,
            self.double_precision,
            False,
        )
        for block in self.transformer_1d_blocks:
            hidden = block(hidden, pe)
        return plain_rms(hidden)
