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
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

CONTENT_MODES = ("image", "image_cond")
IMAGE_ROPE_SCALE = 0.5
ROPE_THETA = 10000
ROPE_AXES = (32, 32, 32, 32)
LATENT_CHANNELS = 128
CONTEXT_DIM = 20480  # 8 Qwen3-VL-4B hidden layers x 2560


@dataclass(frozen=True)
class Flux3Config:
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    fused_qkv: bool = False
    flow_map: bool = False
    mlp_ratio: float = 3.0

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> "Flux3Config":
        """comfy/model_detection.py's FLUX.3 branch."""
        keys = state_dict.keys()
        if "content_mode_blocks.image.0.attn_out.weight" not in keys or "single_stream_modulations.txt.lin.weight" not in keys:
            raise ValueError("Not a FLUX.3 DiT checkpoint (no content_mode_blocks / single_stream_modulations)")
        hidden = state_dict["content_mode_blocks.image.0.attn_out.weight"].shape[0]

        def count(prefix: str) -> int:
            return len({k[len(prefix):].split(".", 1)[0] for k in keys if k.startswith(prefix)})

        return cls(
            hidden_size=hidden,
            num_heads=hidden // 128,
            depth=count("txt_mode_blocks."),
            depth_single_blocks=count("single_blocks."),
            fused_qkv="single_blocks.0.qkv.weight" in keys,
            flow_map="ref_time_in.in_layer.weight" in keys,
        )


# ---------------------------------------------------------------- embeddings and RoPE


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """comfy/ldm/flux/layers.py: the x1000 happens in ``t``'s own dtype."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """[..., n] positions -> [..., n, dim/2, 2, 2] rotation matrices (comfy/ldm/flux/math.py)."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32).to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_from_ids(ids: torch.Tensor) -> torch.Tensor:
    """[N, 4] ids -> [1, 1, N, 64, 2, 2], broadcast over batch and heads of [B, H, N, D]."""
    emb = torch.cat([rope(ids[..., i], ROPE_AXES[i], ROPE_THETA) for i in range(ids.shape[-1])], dim=-3)
    return emb[None, None]


def apply_rope(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    x_ = x.to(pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    out = pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def image_ids(h: int, w: int, index: int, device) -> torch.Tensor:
    ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * IMAGE_ROPE_SCALE
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * IMAGE_ROPE_SCALE
    ids[..., 3] = index
    return ids.flatten(0, 1)


def text_ids(length: int, device) -> torch.Tensor:
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids


# ---------------------------------------------------------------- modules


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Modulation(nn.Module):
    """Single adaLN: (shift, scale, gate), each [B, 1, D]."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.lin(F.silu(vec[:, None, :])).chunk(3, dim=-1)
        return shift, scale, gate


class Block(nn.Module):
    """Parallel attention + SwiGLU MLP under one adaLN and one gate."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, fused_qkv: bool):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        mlp_hidden = int(hidden_size * mlp_ratio)
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """``shift``/``scale``/``gate`` broadcast over x: [B, 1, D] for one mode, [B, N, D] per row."""
        B, N, _ = x.shape
        x_mod = self.pre_norm(x) * (1 + scale) + shift
        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)

        heads = lambda t: t.view(B, N, self.num_heads, -1).transpose(1, 2)  # noqa: E731
        q = apply_rope(self.norm.query_norm(heads(q)), pe)
        k = apply_rope(self.norm.key_norm(heads(k)), pe)
        attn = F.scaled_dot_product_attention(q, k, heads(v))
        out = self.attn_out(attn.transpose(1, 2).reshape(B, N, -1))
        gate_mlp, up = mlp.chunk(2, dim=-1)
        out = out + self.mlp_out(F.silu(gate_mlp) * up)
        return x + out * gate


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec[:, None, :]).chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class Flux3Transformer(nn.Module):
    def __init__(self, config: Flux3Config = Flux3Config()):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.emb_in = nn.ModuleDict({m: nn.Linear(LATENT_CHANNELS, h, bias=False) for m in CONTENT_MODES})
        self.txt_in = nn.Linear(CONTEXT_DIM, h, bias=False)
        self.time_in = MLPEmbedder(256, h)
        if config.flow_map:
            self.ref_time_in = MLPEmbedder(256, h)
        self.early_stream_modulations = nn.ModuleDict({m: Modulation(h) for m in CONTENT_MODES + ("txt",)})
        self.single_stream_modulations = nn.ModuleDict({m: Modulation(h) for m in CONTENT_MODES + ("txt",)})

        def block() -> Block:
            return Block(h, config.num_heads, config.mlp_ratio, config.fused_qkv)

        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([block() for _ in range(config.depth)]) for m in CONTENT_MODES})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(config.depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(config.depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(h, LATENT_CHANNELS)})

        self.gradient_checkpointing = False
        # Activation dtype: weights may be int8, so it cannot be read off a parameter.
        self.compute_dtype = torch.bfloat16

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _run(self, block: Block, x, shift, scale, gate, pe):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, shift, scale, gate, pe, use_reentrant=False)
        return block(x, shift, scale, gate, pe)

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
        for block in self.content_mode_blocks["image"]:
            img = self._run(block, img, shift, scale, gate, pe[:, :, :n_img])

        # joint [image | text]: every row modulated by its own mode
        joint = torch.cat([img, txt], dim=1)
        img_mod = self.single_stream_modulations["image"](vec_img)
        txt_mod = self.single_stream_modulations["txt"](vec_zero)
        n_txt = txt.shape[1]
        shift, scale, gate = (
            torch.cat([i.expand(B, n_img, -1), t.expand(B, n_txt, -1)], dim=1) for i, t in zip(img_mod, txt_mod)
        )
        for block in self.single_blocks:
            joint = self._run(block, joint, shift, scale, gate, pe)

        out = self.final_layer["image"](joint[:, :n_img], vec_img)
        return out.transpose(1, 2).reshape(B, LATENT_CHANNELS, H, W)
