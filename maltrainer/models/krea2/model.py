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

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.krea2.layers import (
    Attention,
    DoubleSharedModulation,
    LastLayer,
    QKNorm,
    RMSNorm,
    SimpleModulation,
    SingleStreamBlock,
    SwiGLU,
    TextFusionBlock,
    TextFusionTransformer,
)
from maltrainer.models.krea2.rope import apply_rope1, embed_nd, rope, timestep_embedding

__all__ = [
    "Attention",
    "DoubleSharedModulation",
    "Krea2Config",
    "Krea2Transformer",
    "LastLayer",
    "QKNorm",
    "RMSNorm",
    "SimpleModulation",
    "SingleStreamBlock",
    "SwiGLU",
    "TextFusionBlock",
    "TextFusionTransformer",
    "apply_rope1",
    "embed_nd",
    "rope",
    "timestep_embedding",
]


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

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
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
        bs, dtype = x.shape[0], self.compute_dtype
        img, imgpos, h, w = self.patchify(x)
        img = self.first(img.to(dtype))

        t = self.tmlp(timestep_embedding(sigma, self.config.tdim).unsqueeze(1).to(dtype))
        tvec = self.tproj(t)
        context = self.txtmlp(self.txtfusion(context.to(dtype)))
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
