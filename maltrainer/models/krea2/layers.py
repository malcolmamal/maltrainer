"""Krea 2 DiT building blocks; the module names match ComfyUI's (see ``model``)."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.krea2.rope import apply_rope1


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
        return cast(torch.Tensor, self.down(F.silu(self.gate(x)) * self.up(x)))


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
        return cast(torch.Tensor, self.wo(out * torch.sigmoid(gate.to(out.dtype))))


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (vec + self.lin.to(vec.dtype)).chunk(6, dim=-1)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = vec + self.lin.to(vec.dtype).unsqueeze(0)
        return out.chunk(2, dim=1)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        return cast(torch.Tensor, x + self.mlp(self.postnorm(x)))


class TextFusionTransformer(nn.Module):
    """Fuses the 12-layer Qwen stack: per-token attention across layers, a learned layer mix, then refiners."""

    def __init__(
        self, n_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(n_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        return cast(torch.Tensor, x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(tvec)
        return cast(torch.Tensor, self.linear((1 + scale) * self.norm(x) + shift))
