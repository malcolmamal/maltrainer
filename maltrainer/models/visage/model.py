"""The Visage-Image DiT: a single-stream transformer over [registers, caption, latent cells] that predicts the
rectified-flow velocity noise - x0. The timestep reaches the blocks through adaLN alone (no pooled caption vector).

Captions are filled with learned per-position pads up to their length bucket, so a caption never depends on what it
was batched with. This port takes one caption length per batch (the trainer runs batch 1), so it needs no key mask.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from maltrainer.models.visage.config import VisageConfig
from maltrainer.models.visage.layers import Block, SinusoidalEmbedding, modulate, spiral_rope_tables


class VisageTransformer(nn.Module):
    def __init__(self, config: VisageConfig) -> None:
        super().__init__()
        self.config = c = config
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(c.time_min_freq, c.time_max_freq, c.noise_embed_dims),
            nn.Linear(c.noise_embed_dims, c.width),
            nn.SiLU(),
            nn.Linear(c.width, c.width),
        )
        self.text_proj = nn.Linear(c.text_dim, c.width)
        self.text_pad_embed = nn.Parameter(torch.randn(c.text_length_buckets[-1], c.width) * 0.02)
        self.text_registers = nn.Parameter(torch.randn(c.n_text_registers, c.width) * 0.02)
        self.patch_embed = nn.Conv2d(c.in_channels, c.width, c.patch_size, stride=c.patch_size)
        self.blocks = nn.ModuleList(Block(c.width, c.mlp_ratio) for _ in range(c.depth))
        self.final_norm = nn.LayerNorm(c.width, elementwise_affine=False)
        final_modulation = nn.Linear(c.width, 2 * c.width)
        self.final_mod = nn.Sequential(nn.SiLU(), final_modulation)
        self.out_proj = nn.Linear(c.width, c.in_channels * c.patch_size**2)
        for layer in (final_modulation, self.out_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def prepare_text(self, text: Tensor) -> Tensor:
        """[B, L, width] caption states -> [B, registers + bucket, width]: registers, caption, pads."""
        batch, length, _ = text.shape
        longest = self.config.text_length_buckets[-1]
        if length > longest:
            raise ValueError(f"caption of {length} tokens exceeds the largest bucket {longest}")
        bucket = next(b for b in self.config.text_length_buckets if b >= length)
        pads = self.text_pad_embed[length:bucket].to(text.dtype).expand(batch, -1, -1)
        registers = self.text_registers.to(text.dtype).expand(batch, -1, -1)
        return torch.cat([registers, text, pads], dim=1)

    def embed_time(self, t: Tensor) -> Tensor:
        sinusoid = self.time_embed[0](t)
        return cast(Tensor, self.time_embed[1:](sinusoid.to(self.patch_embed.weight.dtype)))

    def forward(self, x: Tensor, t: Tensor, text: Tensor) -> Tensor:
        """x: noisy latents [B, C, H, W]; t: [B, 1] in [0, 1], 1 = noise; text: caption states [B, L, text_dim].
        Returns the velocity noise - x0, [B, C, H, W]."""
        p = self.config.patch_size
        grid_h, grid_w = x.shape[-2] // p, x.shape[-1] // p
        cond = self.embed_time(t)
        txt = self.prepare_text(self.text_proj(text))
        img = self.patch_embed(x).flatten(2).transpose(1, 2)
        rope = spiral_rope_tables(grid_h, grid_w, x.device)
        h = torch.cat([txt, img], dim=1)
        for block in self.blocks:
            h = block(h, cond, txt.shape[1], rope)
        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        h = self.out_proj(modulate(self.final_norm(h[:, txt.shape[1] :]), shift, scale))
        h = h.view(x.shape[0], grid_h, grid_w, -1, p, p)
        return cast(Tensor, h.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], -1, grid_h * p, grid_w * p))
