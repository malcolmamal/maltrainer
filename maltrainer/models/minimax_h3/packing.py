"""H3's packed [text | audio | video] sequence: the patching, the two sigma schedules, and the RoPE geometry.

These mirror comfy/ldm/minimax/model.py (_axis_from_sqrt_area / _frame_grid / _video_t_grid / PackedLayout)
exactly. H3 does NOT index video tokens by raw patch index: each axis is normalized by the frame's
sqrt(area), centered, and scaled by 32, and the video segment's temporal origin sits directly after the
text segment. Training on a plain arange() grid puts every token at the wrong RoPE frequency.
"""

from __future__ import annotations

import math

import torch

from maltrainer.models.minimax_h3.model import MiniMaxH3TokenTag


def patchify_video(latent: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """Patchify video/image latents matching ComfyUI / MiniMax H3 layout [B, C, T, H, W] -> [B, N_tokens, C*1*2*2]."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b, t * h * w, c * pt * ph * pw)


def apply_flow_shift(t: torch.Tensor, shift: float = 12.0) -> torch.Tensor:
    """Apply discrete flow shift mapping: t_shifted = (shift * t) / (1 + (shift - 1) * t)."""
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


# H3 denoises its two streams on different schedules: video at flow shift 12, audio at 3
# (ComfyUI's sigma_shift_video / sigma_shift_audio). They advance together, so at inference
# the audio sigma is always the closed-form image of the video sigma below -- the model has
# only ever seen (t_video, t_audio) pairs on that curve. Training keeps that coupling no
# matter how the video sigma itself is sampled: e.g. video sigma 0.5 pairs with audio 0.2.
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0


def remap_sigma(
    sigma: torch.Tensor, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT
) -> torch.Tensor:
    """A sigma on the ``from_shift`` schedule -> the same base-grid point on ``to_shift``.

    Mirrors comfy/ldm/minimax/model.py ``time_shift_sigma``.
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def pack_audio_rows(latent: torch.Tensor) -> torch.Tensor:
    """[B, 32, 2, T] audio latent -> [B, 2*T, 32] channel-major rows (left t0..T-1, then right)."""
    b, c, ch, t = latent.shape
    return latent.permute(0, 2, 3, 1).reshape(b, ch * t, c)


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(latent_h: int, latent_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows, plus the w axis."""
    area = math.sqrt(latent_h * latent_w)
    w_axis = _axis_from_sqrt_area(latent_w, 2, area)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(latent_h, 2, area),
        w_axis,
        indexing="ij",
    )
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def _video_t_grid(latent_t: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t)],
        dtype=torch.float64,
    )
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def build_packed_position_ids(
    text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
) -> torch.Tensor:
    """Position ids for H3's [text | audio | video] packed sequence, as [S, 3] float64.

    Matches ComfyUI's PackedLayout (t2va, no references): text tokens advance along the
    temporal axis; audio and video share the media clock's origin at text_len. Audio rows are
    channel-major stereo -- t advances one unit per latent frame (40 per second), h stays 0,
    and w is pinned to the video grid's first column for the left channel and its last column
    for the right one.
    """
    text_pos = torch.zeros(text_len, 3, dtype=torch.float64)
    text_pos[:, 0] = torch.arange(text_len, dtype=torch.float64)

    frame, w_axis = _frame_grid(latent_h, latent_w)
    audio_pos = torch.zeros(audio_t * 2, 3, dtype=torch.float64)
    if audio_t:
        audio_pos[:, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(2)
        audio_pos[:audio_t, 2] = float(w_axis[0])
        audio_pos[audio_t:, 2] = float(w_axis[-1])

    video_pos = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    video_pos[:, :, 0] = _video_t_grid(latent_t, float(text_len))[:, None]
    video_pos[:, :, 1:] = frame[None]

    return torch.cat([text_pos, audio_pos, video_pos.reshape(-1, 3)], dim=0)


def packed_layout(
    num_text_tokens: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Index/tag/position tensors for one [text | audio | video] packed sequence.

    Everything here is a function of the text length, the latent size and the audio
    length, so the conditional and unconditional passes need SEPARATE layouts whenever
    their prompts tokenize to different lengths -- which they always do when the
    unconditional prompt is empty (a single BOS token).

    Timestep rows: 0 = video sigma (text and video tokens), 1 = audio sigma. The model
    indexes its AdaLN table by ``timestep_index * 3 + modality_tag``, as ComfyUI does.
    """
    num_audio_tokens = audio_t * 2
    num_video_tokens = latent_t * (latent_h // 2) * (latent_w // 2)
    audio_start = num_text_tokens
    video_start = audio_start + num_audio_tokens
    seq_len = video_start + num_video_tokens

    text_indices = torch.arange(0, audio_start, device=device)
    audio_indices = torch.arange(audio_start, video_start, device=device)
    video_indices = torch.arange(video_start, seq_len, device=device)
    token_tags = torch.zeros((seq_len,), device=device, dtype=torch.long)
    token_tags[text_indices] = MiniMaxH3TokenTag.TEXT
    token_tags[audio_indices] = MiniMaxH3TokenTag.AUDIO
    token_tags[video_indices] = MiniMaxH3TokenTag.VIDEO
    timestep_indices = torch.zeros((seq_len,), device=device, dtype=torch.long)
    timestep_indices[audio_indices] = 1

    positions = build_packed_position_ids(num_text_tokens, latent_t, latent_h, latent_w, audio_t)
    return {
        "text_indices": text_indices,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": positions.to(device=device),
    }
