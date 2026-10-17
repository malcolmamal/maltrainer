"""
SupraDiT, the 104.1M text-to-image transformer from SupraLabs/Supra2-IMG.

The architecture matches the published inference script. Sampling integrates
velocity from noise (t=0) to the latent (t=1): z = z + dt * v.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

IMG_SIZE = 256
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class Attention(nn.Module):
    """Multi-head self- or cross-attention."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if self.is_self:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            ctx_len = ctx.shape[1]
            q = self.q(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(ctx).view(batch, ctx_len, 2, self.n_heads, self.head_dim)
            k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.bool()[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, length, channels)
        return self.proj(out)


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, ctx_dim=ctx_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


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
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_model, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(d_model, latent_ch * PATCH * PATCH)

    def _position_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Learned 16x16 positions, bicubic-resized when the latent grid is not 256px."""
        pos = self.pos_embed
        side = int(math.isqrt(pos.shape[1]))
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
        return x.reshape(batch, channels, grid_h * patch, grid_w * patch)


def load_supra2_img(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    freeze: bool = True,
) -> tuple[SupraDiT, dict]:
    """Load Supra2-IMG_fp32.safetensors. Returns the model and sidecar tensors.

    Sidecar holds the stored unconditional embedding (``uncond_text``, ``uncond_mask``)
    and checkpoint metadata such as ``ctx_len``. ``freeze`` keeps the weights fixed for LoRA.
    A finetune leaves them trainable.
    """
    path = Path(path)
    model = SupraDiT()
    weights = load_file(str(path), device="cpu")
    sidecar: dict = {}
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
