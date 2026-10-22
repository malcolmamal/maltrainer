"""Flow-shift schedule and the 3D RoPE frequencies of Qwen-Image-2.1."""

from __future__ import annotations

import math

import torch


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
