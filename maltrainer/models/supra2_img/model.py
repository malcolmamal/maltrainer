"""
SupraDiT, the 104.1M text-to-image transformer from SupraLabs/Supra2-IMG.

The architecture matches the published inference script. Sampling integrates
velocity from noise (t=0) to the latent (t=1): z = z + dt * v.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.supra2_img.layers import (
    Attention,
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    modulate,
)

__all__ = [
    "DEPTH",
    "D_CTX",
    "D_MODEL",
    "LATENT_CH",
    "LATENT_SIZE",
    "MAX_CTX_LEN",
    "MLP_RATIO",
    "NUM_TOKENS",
    "N_HEADS",
    "PATCH",
    "VAE_SCALE",
    "Attention",
    "DiTBlock",
    "FinalLayer",
    "SupraDiT",
    "TimestepEmbedder",
    "load_supra2_img",
    "modulate",
]

LATENT_SIZE = 32
LATENT_CH = 4
PATCH = 2
NUM_TOKENS = (LATENT_SIZE // PATCH) ** 2

D_MODEL = 576
DEPTH = 14
N_HEADS = 9
MLP_RATIO = 4.0
D_CTX = 768
MAX_CTX_LEN = 128
VAE_SCALE = 0.18215


class SupraDiT(nn.Module):
    """~100M parameter DiT for rectified-flow text-to-image."""

    def __init__(
        self,
        latent_ch: int = LATENT_CH,
        d_model: int = D_MODEL,
        depth: int = DEPTH,
        n_heads: int = N_HEADS,
        ctx_dim: int = D_CTX,
        mlp_ratio: float = MLP_RATIO,
        num_tokens: int = NUM_TOKENS,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.patch = PATCH
        self.x_embed = nn.Linear(latent_ch * PATCH * PATCH, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.t_embed = TimestepEmbedder(d_model)
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_model, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(d_model, latent_ch * PATCH * PATCH)

    def _position_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Learned 16x16 positions, bicubic-resized when the latent grid is not 256px."""
        pos = self.pos_embed
        side = math.isqrt(pos.shape[1])
        if grid_h == side and grid_w == side:
            return pos
        weight = pos.reshape(1, side, side, pos.shape[-1]).permute(0, 3, 1, 2).float()
        weight = F.interpolate(weight, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        return weight.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, pos.shape[-1]).to(dtype=pos.dtype)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, _, _ = z.shape
        patch = self.patch
        grid_h, grid_w = z.shape[-2] // patch, z.shape[-1] // patch
        x = z.view(batch, channels, grid_h, patch, grid_w, patch).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch, grid_h * grid_w, channels * patch * patch)
        x = self.x_embed(x) + self._position_embed(grid_h, grid_w)
        conditioning = self.t_embed(t)
        ctx = self.ctx_proj(ctx)
        for block in self.blocks:
            x = block(x, conditioning, ctx, ctx_mask)
        x = self.final(x, conditioning)
        x = x.view(batch, grid_h, grid_w, channels, patch, patch).permute(0, 3, 1, 4, 2, 5)
        return cast(torch.Tensor, x.reshape(batch, channels, grid_h * patch, grid_w * patch))


def load_supra2_img(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    freeze: bool = True,
) -> tuple[SupraDiT, dict[str, Any]]:
    """Load Supra2-IMG_fp32.safetensors. Returns the model and sidecar tensors.

    Sidecar holds the stored unconditional embedding (``uncond_text``, ``uncond_mask``)
    and checkpoint metadata such as ``ctx_len``. ``freeze`` keeps the weights fixed for LoRA.
    A finetune leaves them trainable.
    """
    path = Path(path)
    model = SupraDiT()
    weights = load_file(str(path), device="cpu")
    sidecar: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        sidecar.update({key: int(value) if str(value).isdigit() else value for key, value in meta.items()})
    if "__uncond_text__" in weights:
        sidecar["uncond_text"] = weights.pop("__uncond_text__")
    if "__uncond_mask__" in weights:
        sidecar["uncond_mask"] = weights.pop("__uncond_mask__")
    model.load_state_dict(weights, strict=True)
    model.to(device=device, dtype=dtype)
    if freeze:
        model.requires_grad_(False)
        model.eval()
    else:
        model.requires_grad_(True)
        model.train()
    return model, sidecar
