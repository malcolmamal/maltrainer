"""
Qwen-Image-2.1 Diffusion Transformer (Single-stream DiT).
Supports INT8 ConvRot quantized linear layers and native BF16 inference/training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)

logger = logging.getLogger(__name__)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    """Calculate dynamic shift value based on image token length."""
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = image_seq_len * slope + base_shift - slope * base_seq_len
    return float(mu)


def apply_flow_shift(t: torch.Tensor, mu: float = 0.5) -> torch.Tensor:
    """Apply exponential flow shift mapping: t_shifted = (exp(mu) * t) / (1 + (exp(mu) - 1) * t)."""
    shift = math.exp(mu)
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


class TemporalTimesteps(nn.Module):
    """Sinusoidal timestep embedding generator."""

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.get_buffer("freqs").to(timestep.device)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(timestep.dtype)


class TimestepEmbedder(nn.Module):
    """Timestep MLP projection."""

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 4096):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class TimestepProjEmbeddings(nn.Module):
    """Combined Timestep Projection and Embedding block."""

    def __init__(self, embedding_dim: int = 4096):
        super().__init__()
        self.time_proj = TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedder(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj(timestep)
        return self.timestep_embedder(proj.to(dtype=hidden_states.dtype))


class ZeroCenterRMSNorm(nn.Module):
    """Zero-centered RMSNorm for text projection input."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1.0)).to(input_dtype)


class TextProjection(nn.Module):
    """Text embedding projection layer."""

    def __init__(self, context_in_dim: int = 4096, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU MLP layer."""

    def __init__(self, hidden_size: int = 4096, mlp_hidden_size: int = 12288):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, mlp_hidden_size * 2, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return self.out(F.silu(gate) * up)


class RMSNorm(nn.Module):
    """RMSNorm for attention queries and keys."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=-1, eps=self.eps).to(x.dtype)
        return norm * (self.weight.shape[-1] ** 0.5) * self.weight


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding."""
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # [B, 1, S, D/2]
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    """Multi-head attention module with rotary embeddings and QK-norm."""

    def __init__(self, dim: int = 4096, heads: int = 32, dim_head: int = 128, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(b, s, self.heads, self.dim_head)
        k = self.to_k(hidden_states).view(b, s, self.heads, self.dim_head)
        v = self.to_v(hidden_states).view(b, s, self.heads, self.dim_head)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if rotary_emb is not None:
            q = apply_rotary_emb_qwen(q, rotary_emb)
            k = apply_rotary_emb_qwen(k, rotary_emb)

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, -1)
        out = self.to_out[0](out)
        return out


class TransformerBlock(nn.Module):
    """Single-stream Qwen 2.1 Transformer block."""

    def __init__(self, dim: int = 4096, heads: int = 32, dim_head: int = 128, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        if target_token_mask is None:
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        else:
            real_s, zero_s = scale[:-1].unsqueeze(1), scale[-1:].unsqueeze(0)
            scale = torch.where(target_token_mask.view(1, -1, 1), real_s, zero_s)
            real_g, zero_g = gate[:-1].unsqueeze(1), gate[-1:].unsqueeze(0)
            gate = torch.where(target_token_mask.view(1, -1, 1), real_g, zero_g)
        return hidden_states * (1.0 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_out = self.attn(img_modulated, rotary_emb=rotary_emb, attn_mask=attn_mask)
        hidden_states = hidden_states + img_gate1.tanh() * attn_out

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)
        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm Continuous module."""

    def __init__(self, embedding_dim: int = 4096, conditioning_embedding_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        if target_token_mask is None:
            scale = scale.unsqueeze(1)
        else:
            real_s, zero_s = scale[:-1].unsqueeze(1), scale[-1:].unsqueeze(0)
            scale = torch.where(target_token_mask.view(1, -1, 1), real_s, zero_s)
        return self.norm(hidden_states) * (1.0 + scale)


def build_rope_frequencies(
    text_len: int,
    latent_h: int,
    latent_w: int,
    axes_dim: tuple[int, int, int] = (16, 56, 56),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build 3D RoPE complex frequencies for Qwen-Image-2.1 [text | image] sequence."""
    img_tokens = latent_h * latent_w

    pos_index = torch.arange(8192, device=device)
    neg_index = torch.arange(1024, device=device).flip(0) * -1 - 1

    def rope_p(idx: torch.Tensor, dim: int) -> torch.Tensor:
        freqs = torch.outer(
            idx,
            1.0 / torch.pow(10000.0, torch.arange(0, dim, 2, device=device, dtype=torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    freqs = [torch.cat([rope_p(pos_index, dim), rope_p(neg_index, dim)], dim=0) for dim in axes_dim]

    frame_idx = list(range(text_len)) + [text_len] * img_tokens
    h_idx = list(range(text_len)) + [
        h for h in range(-(latent_h - latent_h // 2), latent_h // 2) for _ in range(latent_w)
    ]
    w_idx = list(range(text_len)) + [
        w for _ in range(latent_h) for w in range(-(latent_w - latent_w // 2), latent_w // 2)
    ]

    rotary_emb = torch.cat(
        [
            freqs[0][torch.tensor(frame_idx, device=device)],
            freqs[1][torch.tensor(h_idx, device=device)],
            freqs[2][torch.tensor(w_idx, device=device)],
        ],
        dim=-1,
    )
    return rotary_emb


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
        return img_pred


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
    if is_int8_convrot(model_path):
        int8_state_dict, num_quant = load_comfy_int8_convrot_state_dict(str(model_path), device=torch.device("cpu"))
    else:
        int8_state_dict, num_quant = load_file(str(model_path), device="cpu"), 0

    with init_empty_weights(include_buffers=False):
        model = QwenImage21Transformer()

    if num_quant > 0:
        logger.info("Configuring %d INT8 ConvRot quantized layers...", num_quant)
        prepare_int8_convrot_modules(model, int8_state_dict)

    model.load_state_dict(int8_state_dict, strict=True, assign=True)

    if num_quant > 0:
        enable_int8_convrot(model, grad_precision=grad_precision)

    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    model.eval()
    logger.info("Loaded Qwen-Image-2.1 Transformer successfully on %s in %s", device, dtype)
    return model
