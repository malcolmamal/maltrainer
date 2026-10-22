"""Clip-length arithmetic of the MiniMax H3 video VAE."""

from __future__ import annotations

# Temporal chunking of the reference VAE (comfy/ldm/minimax/vae.py: clip_length / token_drop).
VIDEO_CLIP_LENGTH = 17
VIDEO_TOKEN_DROP = 3


def align_frame_count(frames: int) -> int:
    """Largest valid H3 clip length <= frames: the 17k + 5 grid (5, 22, 39, 56, ...)."""
    if frames < 5:
        raise ValueError(f"MiniMax H3 clips need at least 5 frames, got {frames}")
    return 5 + ((frames - 5) // 17) * 17


def video_latent_frames(frames: int) -> int:
    """Latent frame count for a 17k + 5 frame clip (comfy_extras/nodes_minimax_h3.py)."""
    return 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2
