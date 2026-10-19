"""
Krea 2 single-stream DiT, ported from ComfyUI (comfy/ldm/krea2/model.py) for training.

Module names match ComfyUI's exactly, so ComfyUI's single-file checkpoints
(krea2_raw_fp8_scaled.safetensors and friends) load by name and LoRA keys come out as
``diffusion_model.blocks.N.attn.wq`` etc., which ComfyUI applies without conversion.

Conventions (ComfyUI ModelType.FLUX, sampling multiplier 1.0):
- the DiT receives sigma in [0, 1] (the sinusoidal embedding multiplies by 1000 itself)
- it predicts the flow velocity v = noise - x0 (denoised = x_t - sigma * v)
- text is a (B, seq, 12, 2560) stack of Qwen3-VL-4B hidden states, fused inside the model
  by ``txtfusion``; text positions are all zero, image tokens sit at (0, row, col)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Krea2Config:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    kvheads: int = 12
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    theta: float = 1e3
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Flux's sinusoidal embedding (cos first), t scaled by 1000."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """Flux RoPE rotation matrices for one axis: [B, L, dim/2, 2, 2] fp32."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def embed_nd(ids: torch.Tensor, axes_dim, theta: float) -> torch.Tensor:
    emb = torch.cat([rope(ids[..., i], axes_dim[i], theta) for i in range(ids.shape[-1])], dim=-3)
    return emb.unsqueeze(1)  # [B, 1, L, headdim/2, 2, 2]


def apply_rope1(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` convention (scale stored zero-centred)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale.float() + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """GQA attention with per-head QK RMSNorm and a sigmoid output gate."""

    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads or heads
        self.headdim = dim // heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, seq, _ = x.shape
        q = self.wq(x).view(b, seq, self.heads, self.headdim).transpose(1, 2)
        k = self.wk(x).view(b, seq, self.kvheads, self.headdim).transpose(1, 2)
        v = self.wv(x).view(b, seq, self.kvheads, self.headdim).transpose(1, 2)
        gate = self.gate(x)
        q, k = self.qknorm.qnorm(q), self.qknorm.knorm(k)
        if freqs is not None:
            q, k = apply_rope1(q, freqs), apply_rope1(k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, seq, -1)
        return self.wo(out * torch.sigmoid(gate.to(out.dtype)))


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: torch.Tensor):
        return (vec + self.lin.to(vec.dtype)).chunk(6, dim=-1)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: torch.Tensor):
        out = vec + self.lin.to(vec.dtype).unsqueeze(0)
        return out.chunk(2, dim=1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask=mask)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    """Fuses the 12-layer Qwen stack: per-token attention across layers, a learned layer mix, then refiners."""

    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x, mask=None):
        b, seq, n, d = x.shape
        x = x.reshape(b * seq, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = x.reshape(b, seq, n, d).permute(0, 1, 3, 2)  # (b seq) n d -> b seq d n
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class Krea2Transformer(nn.Module):
    def __init__(self, config: Krea2Config | None = None):
        super().__init__()
        c = self.config = config or Krea2Config()
        headdim = c.features // c.heads
        self.axes_dim = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        self.first = nn.Linear(c.channels * c.patch**2, c.features, bias=True)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(c.features, c.heads, c.multiplier, False, c.kvheads) for _ in range(c.layers)]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(c.tdim, c.features), nn.GELU(approximate="tanh"), nn.Linear(c.features, c.features)
        )
        self.txtfusion = TextFusionTransformer(c.txtlayers, c.txtdim, c.txtheads, c.multiplier, False, c.txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(c.txtdim),
            nn.Linear(c.txtdim, c.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(c.features, c.features),
        )
        self.last = LastLayer(c.features, c.patch, c.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(c.features, c.features * 6))
        self.gradient_checkpointing = False
        # Activation dtype. Weights may be fp8 (dequantized per call), so it cannot be read
        # off a parameter.
        self.compute_dtype = torch.bfloat16

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def patchify(self, x: torch.Tensor):
        """[B, C, H, W] latent -> ([B, h*w, C*p*p] tokens, [B, h*w, 3] positions, h, w)."""
        p = self.config.patch
        b, c, H, W = x.shape
        h, w = H // p, W // p
        tokens = x.reshape(b, c, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w, c * p * p)
        ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return tokens, ids.reshape(1, h * w, 3).expand(b, -1, -1), h, w

    def unpatchify(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        p, c = self.config.patch, self.config.channels
        b = tokens.shape[0]
        return tokens.reshape(b, h, w, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h * p, w * p)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, 16, H, W] noisy latent; sigma: [B]; context: [B, seq, 12, 2560]. Returns v: [B, 16, H, W].

        No text attention mask, as in ComfyUI: every sample carries its own unpadded prompt.
        """
        bs = x.shape[0]
        dtype = self.compute_dtype
        img, imgpos, h, w = self.patchify(x)
        img = self.first(img.to(dtype))

        t = self.tmlp(timestep_embedding(sigma, self.config.tdim).unsqueeze(1).to(dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context.to(dtype))
        context = self.txtmlp(context)
        txtlen = context.shape[1]

        txtpos = torch.zeros(bs, txtlen, 3, device=x.device, dtype=torch.float32)
        freqs = embed_nd(torch.cat((txtpos, imgpos), dim=1), self.axes_dim, self.config.theta)

        combined = torch.cat((context, img), dim=1)
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                combined = checkpoint(block, combined, tvec, freqs, use_reentrant=False)
            else:
                combined = block(combined, tvec, freqs)

        out = self.last(combined, t)[:, txtlen:]
        return self.unpatchify(out, h, w)
