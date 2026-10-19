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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

ROPE_THETA = 10000
ROPE_AXES = 4
_T = torch.Tensor
LATENT_CHANNELS = 128
TIME_FREQ_DIM = 256
TIME_FACTOR = 1000.0


@dataclass(frozen=True)
class MOSSImageConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    fused_qkv: bool = False

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> MOSSImageConfig:
        """comfy/model_detection.py's MOSS-Image branch."""
        keys = state_dict.keys()
        if "shared_block_modulation.weight" not in keys or "layers.0.proj.ffn_up_gate.weight" not in keys:
            raise ValueError("Not a MOSS-Image DiT checkpoint (no shared_block_modulation / layers.0.proj.ffn_up_gate)")
        dim = state_dict["shared_block_modulation.weight"].shape[0] // 6
        return cls(
            dim=dim,
            ffn_dim=state_dict["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
            num_heads=dim // 128,
            num_layers=len({k.split(".", 2)[1] for k in keys if k.startswith("layers.")}),
            text_dim=state_dict["text_embedder.proj.weight"].shape[1],
            fused_qkv="layers.0.proj.qkv.weight" in keys,
        )


# ---------------------------------------------------------------- embeddings and RoPE


def timestep_embedding(sigma: torch.Tensor, dim: int = TIME_FREQ_DIM, max_period: int = 10000) -> torch.Tensor:
    """[B] sigma -> [B, dim] float32 (cos | sin) of 1000 * sigma, the phases in float64."""
    half = dim // 2
    freqs = max_period ** (-torch.arange(half, device=sigma.device, dtype=torch.float64) / half)
    phase = (TIME_FACTOR * sigma.double())[:, None] * freqs
    return torch.cat([phase.cos(), phase.sin()], dim=-1).float()


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """[n] positions -> [n, dim/2, 2, 2] rotation matrices (comfy/ldm/flux/math.py)."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32).to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def position_ids(h: int, w: int, text_len: int, device: torch.device) -> torch.Tensor:
    """[h*w + text_len, 4] (r, f, h, w): the image first, at f = text_len, rows/cols centred; then text."""
    ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
    ids[: h * w, 1] = text_len
    ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    ids[h * w :, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    return ids


def rope_from_ids(ids: torch.Tensor, head_dim: int) -> torch.Tensor:
    """[N, 4] ids -> [1, N, 1, head_dim/2, 2, 2], broadcast over batch and heads of [B, N, H, D]."""
    axis = head_dim // ROPE_AXES
    emb = torch.cat([rope(ids[:, i], axis, ROPE_THETA) for i in range(ROPE_AXES)], dim=-3)
    return emb[None, :, None]


def apply_rope(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE on [B, N, H, D]: pairs (x[i], x[i + D/2]) (ComfyUI's rope_split_half)."""
    x_ = x.to(pe.dtype).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    out = pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]
    return out.movedim(-1, -2).flatten(-2).type_as(x)


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


# ---------------------------------------------------------------- modules


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def attention(q: _T, k: _T, v: _T, q_norm: RMSNorm, k_norm: RMSNorm, pe: _T, heads: int) -> torch.Tensor:
    """[B, N, H*D] q/k/v -> [B, N, H*D]. QK-norm and RoPE in fp32, rounded once, as ComfyUI's
    fused rms_rope_split_half kernel does at inference."""
    B, N, _ = q.shape
    q = apply_rope(q_norm(q.view(B, N, heads, -1).float()), pe).to(v.dtype)
    k = apply_rope(k_norm(k.view(B, N, heads, -1).float()), pe).to(v.dtype)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.view(B, N, heads, -1).transpose(1, 2))
    return out.transpose(1, 2).reshape(B, N, -1)


class TextBlock(nn.Module):
    """Unmodulated pre-norm block of the text embedder; its Linears carry biases."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, fused_qkv: bool):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({n: nn.Linear(dim, 3 * dim if n == "qkv" else dim, bias=False) for n in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self, x: _T, shift_a: _T, scale_a: _T, gate_a: _T, shift_f: _T, scale_f: _T, gate_f: _T, pe: _T
    ) -> torch.Tensor:
        """The six adaLN vectors are [B, 1, D], shared by every token."""
        y = self.norm1(x) * (1 + scale_a) + shift_a
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        x = x + self.proj["o"](attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads)) * gate_a
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f


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
        return out.transpose(1, 2).reshape(B, LATENT_CHANNELS, H, W)
