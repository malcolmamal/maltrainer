"""
MOSS-Image DiT, written for LoRA training.

Mirrors comfy/ldm/moss_image/model.py. Its module names match ComfyUI's exactly, so ComfyUI's
single files (moss_image_base_bf16 / _int8_convrot) load with ``strict=True``. LoRA keys come out
as ``diffusion_model.layers.N.proj.q`` etc., which ComfyUI applies without conversion, also to
the repacked files with a fused ``proj.qkv`` (comfy/lora.py slices them in).

Architecture (5B, OpenMOSS technical report, section A.1):
- text: Qwen3.5-9B's last hidden state -> ``text_embedder``: a Linear (4096 -> 3072) and two
  unmodulated pre-norm transformer blocks over the text alone, with their own fused qkv and a
  SiTU feed-forward
- image: the Flux2 VAE's 128-channel latent, one token per 16 px (no further patching), through
  one Linear (``video_embedder``)
- 42 single-stream blocks over the joint ``[image | text]`` sequence. Every block and every token,
  text included, takes the same six adaLN vectors from one ``shared_block_modulation`` of the
  timestep embedding. Attention with QK-norm, then a SiTU feed-forward, each under its own gate.
- SiTU: ``(4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25))``, a soft-capped gated activation
- 4-axis RoPE (r, f, h, w), 32 dims each, theta 10000, split-half pairs. r is always 0 (no
  reference images), so the first quarter of every head is unrotated. Text advances along f; the
  image sits at f = text length with rows and columns centred on zero.

Conventions (the report's equation 3, ComfyUI ModelType.FLOW):
- the DiT gets sigma in [0, 1]; the sinusoidal embedding is of 1000 * sigma, computed in float64
  and fed to the MLP in float32, as the training code does (OpenMOSS's ComfyUI node)
- it predicts the flow velocity v = noise - x0

The pieces live in ``model_config`` (the config), ``model_math`` (timestep embedding, RoPE, SiTU) and
``model_blocks`` (the modules the transformer is made of); every name is re-exported here.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.moss_image.model_blocks import _T, Block, RMSNorm, TextBlock, TextEmbedder, attention
from maltrainer.models.moss_image.model_config import MOSSImageConfig
from maltrainer.models.moss_image.model_math import (
    LATENT_CHANNELS,
    ROPE_AXES,
    ROPE_THETA,
    TIME_FACTOR,
    TIME_FREQ_DIM,
    apply_rope,
    position_ids,
    rope,
    rope_from_ids,
    situ,
    timestep_embedding,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LATENT_CHANNELS",
    "ROPE_AXES",
    "ROPE_THETA",
    "TIME_FACTOR",
    "TIME_FREQ_DIM",
    "Block",
    "MOSSImageConfig",
    "MOSSImageTransformer",
    "RMSNorm",
    "TextBlock",
    "TextEmbedder",
    "apply_rope",
    "attention",
    "position_ids",
    "rope",
    "rope_from_ids",
    "situ",
    "timestep_embedding",
]


class MOSSImageTransformer(nn.Module):
    def __init__(self, config: MOSSImageConfig | None = None):
        super().__init__()
        self.config = config = config or MOSSImageConfig()
        d = config.dim
        self.head_dim = d // config.num_heads
        self.video_embedder = nn.Linear(LATENT_CHANNELS, d)
        self.text_embedder = TextEmbedder(config.text_dim, d, config.ffn_dim, config.num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(TIME_FREQ_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.shared_block_modulation = nn.Linear(d, 6 * d)
        self.layers = nn.ModuleList(
            [Block(d, config.ffn_dim, config.num_heads, config.fused_qkv) for _ in range(config.num_layers)]
        )
        self.final_norm_video = RMSNorm(d)
        self.final_linear_video = nn.Linear(d, LATENT_CHANNELS, bias=False)
        self.video_head_modulation = nn.Linear(d, 2 * d)

        self.gradient_checkpointing = False
        # Activation dtype: weights may be int8, so it cannot be read off a parameter.
        self.compute_dtype = torch.bfloat16

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _run(self, block: nn.Module, *args: Any) -> Any:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, x: _T, sigma: _T, context: _T, skip_layers: tuple[int, ...] = ()) -> torch.Tensor:
        """x [B, 128, H, W] noisy latent, sigma [B] in [0, 1], context [B, L, 4096] -> v [B, 128, H, W].

        skip_layers: joint blocks to leave out (CFG's negative branch skips block 22 in the reference
        sampler; ComfyUI's SkipLayerGuidanceDiTSimple)."""
        B, _, H, W = x.shape
        dtype, device = self.compute_dtype, x.device
        x, context = x.to(dtype), context.to(dtype)
        if context.shape[0] != B:
            context = context.expand(B, -1, -1)
        n_img, n_txt = H * W, context.shape[1]
        pe = rope_from_ids(position_ids(H, W, n_txt, device), self.head_dim)

        vec = self.time_embedding(timestep_embedding(sigma.reshape(-1).expand(B)).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        txt = self.text_embedder.proj(context)
        txt_pe = pe[:, n_img:]
        for block in self.text_embedder.blocks:
            txt = self._run(block, txt, txt_pe)

        joint = torch.cat([self.video_embedder(x.flatten(2).transpose(1, 2)), txt], dim=1)
        for index, block in enumerate(self.layers):
            if index in skip_layers:
                continue
            joint = self._run(block, joint, *mod, pe)

        out = self.final_norm_video(joint[:, :n_img]) * (1 + head_scale) + head_shift
        out = self.final_linear_video(out)
        return cast(torch.Tensor, out.transpose(1, 2).reshape(B, LATENT_CHANNELS, H, W))
