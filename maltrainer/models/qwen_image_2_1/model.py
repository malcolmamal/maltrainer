"""
Qwen-Image-2.1 Diffusion Transformer (Single-stream DiT).
Supports INT8 ConvRot quantized linear layers and native BF16 inference/training.

The building blocks live in ``model_layers`` and ``model_blocks``, the schedule and RoPE maths in ``model_math``;
every public name is re-exported here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
from accelerate import init_empty_weights
from torch import nn

from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    prepare_int8_convrot_modules,
)
from maltrainer.models.qwen_image_2_1.checkpoint import read_state_dict, to_keeping_scales
from maltrainer.models.qwen_image_2_1.model_blocks import (
    AdaLayerNormContinuous,
    Attention,
    TransformerBlock,
)
from maltrainer.models.qwen_image_2_1.model_layers import (
    RMSNorm,
    SwiGLUFeedForward,
    TemporalTimesteps,
    TextProjection,
    TimestepEmbedder,
    TimestepProjEmbeddings,
    ZeroCenterRMSNorm,
    apply_rotary_emb_qwen,
)
from maltrainer.models.qwen_image_2_1.model_math import apply_flow_shift, build_rope_frequencies, calculate_shift

logger = logging.getLogger(__name__)

__all__ = [
    "AdaLayerNormContinuous",
    "Attention",
    "QwenImage21Transformer",
    "RMSNorm",
    "SwiGLUFeedForward",
    "TemporalTimesteps",
    "TextProjection",
    "TimestepEmbedder",
    "TimestepProjEmbeddings",
    "TransformerBlock",
    "ZeroCenterRMSNorm",
    "apply_flow_shift",
    "apply_rotary_emb_qwen",
    "build_rope_frequencies",
    "calculate_shift",
    "load_qwen_image_2_1_transformer",
]


class QwenImage21Transformer(nn.Module):
    """Qwen-Image-2.1 Single-Stream Diffusion Transformer."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        dim: int = 4096,
        heads: int = 32,
        dim_head: int = 128,
        mlp_ratio: int = 3,
        context_in_dim: int = 4096,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.inner_dim = dim
        self.out_channels = out_channels
        self.axes_dims_rope = axes_dims_rope

        self.time_text_embed = TimestepProjEmbeddings(embedding_dim=dim)
        self.txt_in = TextProjection(context_in_dim, dim, eps=eps)
        self.img_in = nn.Linear(in_channels, dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio, eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(dim, dim, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Forward pass of Qwen-Image-2.1 transformer.

        Args:
            hidden_states: Image latents [B, H*W, 64] (or [B, 64, H, W])
            encoder_hidden_states: Text embeddings [B, text_len, 4096]
            timestep: Normalized timesteps in [0.0, 1.0] shape [B]
            latent_h: Spatial height in latent tokens
            latent_w: Spatial width in latent tokens
        Returns:
            Velocity prediction [B, H*W, 64]
        """
        b = hidden_states.shape[0]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(b, latent_h * latent_w, self.out_channels)

        text_len = encoder_hidden_states.shape[1]
        img_tokens = latent_h * latent_w
        seq_len = text_len + img_tokens

        img_feats = self.img_in(hidden_states)
        txt_feats = self.txt_in(encoder_hidden_states)
        joint = torch.cat([txt_feats, img_feats], dim=1)

        target_token_mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
        target_token_mask[text_len:] = True

        # Conditioning timestep: real for image, 0 for text
        ts_cat = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        temb = self.time_text_embed(ts_cat, joint)
        modulation = self.modulation(temb)

        # Block-causal attention mask: text is causal (cannot see img), img attends to everything
        q_idx = torch.arange(seq_len, device=hidden_states.device)[:, None]
        kv_idx = torch.arange(seq_len, device=hidden_states.device)[None, :]
        attn_mask = (q_idx >= kv_idx) | (q_idx >= text_len)
        attn_mask = attn_mask & ~((q_idx < text_len) & (kv_idx >= text_len))

        # RoPE frequencies
        rotary_emb = build_rope_frequencies(
            text_len=text_len,
            latent_h=latent_h,
            latent_w=latent_w,
            axes_dim=self.axes_dims_rope,
            device=hidden_states.device,
        )

        hidden = joint
        for block in self.transformer_blocks:
            hidden = block(
                hidden,
                modulation,
                rotary_emb=rotary_emb,
                attn_mask=attn_mask,
                target_token_mask=target_token_mask,
            )

        hidden = self.norm_out(hidden, temb, target_token_mask)
        pred = self.proj_out(hidden)
        img_pred = pred[:, text_len:]
        return cast(torch.Tensor, img_pred)


def load_qwen_image_2_1_transformer(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    grad_precision: str = "bf16",
) -> QwenImage21Transformer:
    """Load Qwen-Image-2.1 Transformer from INT8 ConvRot or BF16 safetensors checkpoint."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Transformer model file not found at {model_path}")

    logger.info("Loading Qwen-Image-2.1 Transformer from %s...", model_path)
    int8_state_dict, num_quant = read_state_dict(str(model_path))

    with init_empty_weights(include_buffers=False):
        model = QwenImage21Transformer()

    if num_quant > 0:
        logger.info("Configuring %d INT8 ConvRot quantized layers...", num_quant)
        prepare_int8_convrot_modules(model, int8_state_dict)

    model.load_state_dict(int8_state_dict, strict=True, assign=True)

    if num_quant > 0:
        enable_int8_convrot(model, grad_precision=grad_precision)

    model.requires_grad_(False)
    to_keeping_scales(model, device, dtype)
    model.eval()
    logger.info("Loaded Qwen-Image-2.1 Transformer successfully on %s in %s", device, dtype)
    return model
