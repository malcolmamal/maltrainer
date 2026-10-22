"""
FLUX.3 [klein] DiT, written for LoRA training.

Mirrors comfy/ldm/flux3/model.py, and its module names match ComfyUI's exactly. ComfyUI's
single-file checkpoints (flux3_image_klein_bf16 / _int8_convrot) therefore load with
``strict=True``, and LoRA keys come out as ``diffusion_model.single_blocks.N.q_proj`` etc.,
which ComfyUI applies without conversion. It also applies them to files with a fused ``qkv``.

Architecture (text-to-image path; reference images, the ``image_cond`` mode, are not trained):
- three input modes, ``image`` / ``image_cond`` / ``txt``, each with its own 5 "early" blocks and
  its own adaLN modulations. Text runs its early blocks alone at t = 0, and so does the image at
  the real t.
- then 28 single blocks over the joint ``[image | text]`` sequence. Every row is modulated by
  its own mode's adaLN: the image by t, the text by t = 0.
- each block is parallel attention + SwiGLU MLP under one pre-norm, summed into one gate
- 4-axis RoPE (t, row, col, l), 32 dims each, theta 10000. Image rows and columns are scaled by
  0.5. Text advances along l.

Conventions (ComfyUI ModelType.FLUX, the diffusers FLUX.3 trainer):
- the DiT gets sigma in [0, 1]. The sinusoidal embedding multiplies it by 1000, *in the compute
  dtype*, as the reference does, so a bf16 run embeds a bf16-rounded timestep.
- it predicts the flow velocity v = noise - x0
- latents are the Flux2 VAE's 128-channel, 16x-downsampled, batch-normed latents. One token per
  latent pixel, no further patching.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from maltrainer.models.flux3.config import Flux3Config
from maltrainer.models.flux3.layers import Block, LastLayer, MLPEmbedder, Modulation, QKNorm, RMSNorm
from maltrainer.models.flux3.rope import (
    IMAGE_ROPE_SCALE,
    ROPE_AXES,
    ROPE_THETA,
    apply_rope,
    image_ids,
    rope,
    rope_from_ids,
    text_ids,
    timestep_embedding,
)

logger = logging.getLogger(__name__)

CONTENT_MODES = ("image", "image_cond")
LATENT_CHANNELS = 128
CONTEXT_DIM = 20480  # 8 Qwen3-VL-4B hidden layers x 2560

__all__ = [
    "CONTENT_MODES",
    "CONTEXT_DIM",
    "IMAGE_ROPE_SCALE",
    "LATENT_CHANNELS",
    "ROPE_AXES",
    "ROPE_THETA",
    "Block",
    "Flux3Config",
    "Flux3Transformer",
    "LastLayer",
    "MLPEmbedder",
    "Modulation",
    "QKNorm",
    "RMSNorm",
    "apply_rope",
    "image_ids",
    "rope",
    "rope_from_ids",
    "text_ids",
    "timestep_embedding",
]


class Flux3Transformer(nn.Module):
    def __init__(self, config: Flux3Config | None = None):
        super().__init__()
        self.config = config = config or Flux3Config()
        h = config.hidden_size
        self.emb_in = nn.ModuleDict({m: nn.Linear(LATENT_CHANNELS, h, bias=False) for m in CONTENT_MODES})
        self.txt_in = nn.Linear(CONTEXT_DIM, h, bias=False)
        self.time_in = MLPEmbedder(256, h)
        if config.flow_map:
            self.ref_time_in = MLPEmbedder(256, h)
        self.early_stream_modulations = nn.ModuleDict({m: Modulation(h) for m in (*CONTENT_MODES, "txt")})
        self.single_stream_modulations = nn.ModuleDict({m: Modulation(h) for m in (*CONTENT_MODES, "txt")})

        def block() -> Block:
            return Block(h, config.num_heads, config.mlp_ratio, config.fused_qkv)

        self.content_mode_blocks = nn.ModuleDict(
            {m: nn.ModuleList([block() for _ in range(config.depth)]) for m in CONTENT_MODES}
        )
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(config.depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(config.depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(h, LATENT_CHANNELS)})
        self.gradient_checkpointing = False
        # Activation dtype: weights may be int8, so it cannot be read off a parameter.
        self.compute_dtype = torch.bfloat16

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _run(self, block: nn.Module, x: Tensor, shift: Tensor, scale: Tensor, gate: Tensor, pe: Tensor) -> Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return cast(Tensor, checkpoint(block, x, shift, scale, gate, pe, use_reentrant=False))
        return cast(Tensor, block(x, shift, scale, gate, pe))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x [B, 128, H, W] noisy latent, sigma [B] in [0, 1], context [B, L, 20480] -> v [B, 128, H, W]."""
        B, _, H, W = x.shape
        dtype, device = self.compute_dtype, x.device
        x = x.to(dtype)
        context = context.to(dtype)

        # image at its own t; text (and references) at t = 0
        vec = self.time_in(timestep_embedding(torch.cat([sigma.to(dtype), sigma.new_zeros(1, dtype=dtype)]), 256))
        vec_img, vec_zero = vec[:-1], vec[-1:]

        # text: its 5 early blocks alone, at t = 0 (ComfyUI's preprocess_conds)
        txt = self.txt_in(context)
        if txt.shape[0] != B:
            txt = txt.expand(B, -1, -1)
        txt_pe = rope_from_ids(text_ids(txt.shape[1], device))
        shift, scale, gate = self.early_stream_modulations["txt"](vec_zero)
        for block in self.txt_mode_blocks:
            txt = self._run(block, txt, shift, scale, gate, txt_pe)

        # image: its 5 early blocks alone, at t
        img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
        img_ids = image_ids(H, W, 0, device)
        pe = rope_from_ids(torch.cat([img_ids, text_ids(txt.shape[1], device)]))
        n_img = H * W
        shift, scale, gate = self.early_stream_modulations["image"](vec_img)
        for block in cast("nn.ModuleList", self.content_mode_blocks["image"]):
            img = self._run(block, img, shift, scale, gate, pe[:, :, :n_img])

        # joint [image | text]: every row modulated by its own mode
        joint = torch.cat([img, txt], dim=1)
        img_mod = self.single_stream_modulations["image"](vec_img)
        txt_mod = self.single_stream_modulations["txt"](vec_zero)
        n_txt, pairs = txt.shape[1], zip(img_mod, txt_mod, strict=False)
        shift, scale, gate = (torch.cat([i.expand(B, n_img, -1), t.expand(B, n_txt, -1)], dim=1) for i, t in pairs)
        for block in self.single_blocks:
            joint = self._run(block, joint, shift, scale, gate, pe)
        out = self.final_layer["image"](joint[:, :n_img], vec_img)
        return cast(torch.Tensor, out.transpose(1, 2).reshape(B, LATENT_CHANNELS, H, W))
