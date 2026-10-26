"""The Visage-Image VAE decoder: a ViT on the 16x16 latent grid, each token split 2x2 into a second ViT stage on the
f8 grid, an 8x8 linear patch head, and two residual conv refiners that work across the patch seams."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from maltrainer.models.visage.vae_config import VAEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).type_as(x)


def axial_rope_tables(head_dim: int, grid: int, device: torch.device, theta: float = 100.0) -> tuple[Tensor, Tensor]:
    quarter = head_dim // 4
    freqs = 1.0 / (theta ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    angles = torch.arange(grid, dtype=torch.float32)[:, None] * freqs[None, :]
    rows = angles[:, None, :].expand(grid, grid, quarter)
    cols = angles[None, :, :].expand(grid, grid, quarter)
    table = torch.cat([rows, cols], dim=-1).reshape(grid * grid, head_dim // 2)
    return table.cos().to(device), table.sin().to(device)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, n, w = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        return cast(Tensor, self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w)))


class SwiGLU(nn.Module):
    def __init__(self, width: int, multiple: int = 64) -> None:
        super().__init__()
        hidden = multiple * ((int(8 * width / 3) + multiple - 1) // multiple)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, layer_scale: float) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(width), Attention(width, heads)
        self.norm2, self.mlp = RMSNorm(width), SwiGLU(width)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(width))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(width))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x), cos, sin)
        return cast(Tensor, x + self.gamma2 * self.mlp(self.norm2(x)))


def refiner(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, channels, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(channels, 3, 3, padding=1),
    )


class ViTDecoder(nn.Module):
    def __init__(self, c: VAEConfig) -> None:
        super().__init__()
        self.from_latent = nn.Linear(c.latent_channels, c.decoder_width)
        self.blocks = nn.ModuleList(
            Block(c.decoder_width, c.decoder_heads, c.layer_scale) for _ in range(c.decoder_depth)
        )
        self.norm = RMSNorm(c.decoder_width)
        self.to_hires = nn.Linear(c.decoder_width, 4 * c.hires_width)  # each f16 token into its 2x2 f8 children
        self.hires_blocks = nn.ModuleList(
            Block(c.hires_width, c.hires_heads, c.layer_scale) for _ in range(c.hires_depth)
        )
        self.hires_norm = RMSNorm(c.hires_width)
        self.head = nn.Linear(c.hires_width, 8 * 8 * 3)
        self.refiner = refiner(c.refiner_width)
        self.refiner2 = refiner(c.refiner2_width)
        self.head_dims = (c.decoder_width // c.decoder_heads, c.hires_width // c.hires_heads)

    def forward(self, latent: Tensor) -> Tensor:
        """Raw latents [B, C, h, h] -> images [B, 3, 16h, 16h] in [-1, 1]."""
        b, _, h, w = latent.shape
        if h != w:
            raise ValueError("latent grid must be square")
        x = self.from_latent(latent.flatten(2).transpose(1, 2))
        cos, sin = axial_rope_tables(self.head_dims[0], h, latent.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.to_hires(self.norm(x)).transpose(1, 2).reshape(b, -1, h, w)
        x = F.pixel_shuffle(x, 2).flatten(2).transpose(1, 2)
        cos, sin = axial_rope_tables(self.head_dims[1], 2 * h, latent.device)
        for hires in self.hires_blocks:
            x = hires(x, cos, sin)
        x = self.head(self.hires_norm(x)).transpose(1, 2).reshape(b, 3 * 64, 2 * h, 2 * w)
        image = F.pixel_shuffle(x, 8)
        image = image + self.refiner(image)
        image = image + self.refiner2(image)
        return torch.tanh(image)
