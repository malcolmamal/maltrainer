"""
MiniMax H3 audio VAE -- encoder side only.

A DAC-lineage waveform encoder followed by an attention projection down to 32 latent
channels. Training only ever needs ``encode``, so the BigVGAN decoder is not built and its
tensors are skipped on load.

Module layout mirrors comfy/ldm/minimax/audio_vae.py exactly, so ComfyUI's single-file
``minimax_h3_audio_vae_fp32.safetensors`` loads by name. The original release
(``FL2VA/audio_vae/model.safetensors``) stores weight-norm ``weight_g``/``weight_v`` pairs
instead of plain weights; those are folded on load.

Shapes (ComfyUI convention):
    waveform [B, 2, L] at 32 kHz, in [-1, 1]  ->  normalized latents [B, 32, 2, T]
    T = ceil(L / 800), i.e. 40 latent frames per second. Each stereo channel goes through
    the mono encoder as its own batch item.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

SAMPLE_RATE = 32000
HOP_LENGTH = 800  # prod(encoder strides)
LATENTS_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 40
LATENT_CHANNELS = 32

# Per-channel latent statistics from the released FL2VA/audio_vae/config.json. ComfyUI's
# single file carries them as buffers; these are only the fallback for the original release.
DEFAULT_LATENTS_MEAN = (
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797,
)
DEFAULT_LATENTS_STD = (
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363,
)


def _snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # the snake activation: x plus sin squared of (alpha * x), scaled by 1 / beta
    return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(x.dtype)
        return _snake(x, alpha, alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return cast(torch.Tensor, y + x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class DacEncoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: tuple[int, ...] = (2, 4, 4, 5, 5), d_latent: int = 2048):
        super().__init__()
        block: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return cast(torch.Tensor, self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        bias = torch.cat((self.q_bias, self.get_buffer("zero_k_bias"), self.v_bias)).to(x.dtype)
        qkv = F.linear(x, self.qkv.weight.to(x.dtype), bias)
        q, k, v = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # mean over heads, then pool the width down to the latent width
        x = F.adaptive_avg_pool1d(x.mean(dim=1), self.out_dim)
        return cast(torch.Tensor, self.proj(x))


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return cast(torch.Tensor, x + self.mlp(self.norm2(x)))


class MiniMaxH3AudioEncoder(nn.Module):
    """Stereo 32 kHz waveform -> normalized H3 audio latents. Frozen, fp32."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DacEncoder(64, (2, 4, 4, 5, 5), 2048)
        self.pre_block = AttnProjection(2048, LATENT_CHANNELS, num_heads=8)
        self.mean_proj = nn.Conv1d(LATENT_CHANNELS, LATENT_CHANNELS, 1)
        self.register_buffer("latents_mean", torch.tensor(DEFAULT_LATENTS_MEAN, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(DEFAULT_LATENTS_STD, dtype=torch.float32))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, 2, L] at 32 kHz -> normalized latents [B, 32, 2, ceil(L / 800)].

        The tail is zero-padded to a whole latent frame and the posterior mean is used
        (no sampling), as in ComfyUI and the released pipeline.
        """
        b, s, length = waveform.shape
        waveform = waveform.float()
        right_pad = math.ceil(length / HOP_LENGTH) * HOP_LENGTH - length
        x = F.pad(waveform, (0, right_pad)).reshape(b * s, 1, -1)
        x = self.encoder(x)
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        z = (z - self.get_buffer("latents_mean").view(1, -1, 1)) / self.get_buffer("latents_std").view(1, -1, 1)
        return cast(torch.Tensor, z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3))


def _fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold weight_g/weight_v pairs (nn.utils.weight_norm, dim=0) into plain weights."""
    folded = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            stem = name[: -len("weight_v")]
            g = state_dict[stem + "weight_g"]
            norm = tensor.norm(p=2, dim=tuple(range(1, tensor.ndim)), keepdim=True)
            folded[stem + "weight"] = g * tensor / norm
        else:
            folded[name] = tensor
    return folded


_ENCODER_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "latents_mean", "latents_std")


def load_minimax_h3_audio_encoder(path: str | Path, device: torch.device) -> MiniMaxH3AudioEncoder:
    """Load the audio VAE encoder from ComfyUI's single file or the original release."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    if any(key.endswith(".weight_g") for key in state):
        state = _fold_weight_norm(state)
    state = {key: value.float() for key, value in state.items() if key.startswith(_ENCODER_PREFIXES)}
    model = MiniMaxH3AudioEncoder()
    has_stats = "latents_mean" in state and "latents_std" in state
    info = model.load_state_dict(state, strict=False)
    missing = [key for key in info.missing_keys if key not in ("latents_mean", "latents_std")]
    if missing:
        raise ValueError(
            f"{Path(path).name} did not supply {len(missing)} audio VAE encoder tensor(s); "
            f"examples: {missing[:5]}. Is it the MiniMax H3 audio VAE?"
        )
    if info.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} has {len(info.unexpected_keys)} encoder tensor(s) this module does "
            f"not expect; examples: {info.unexpected_keys[:5]}"
        )
    if not has_stats:
        logger.info("%s carries no latent statistics; using the released config values.", Path(path).name)

    # fp32 only: the reference pins this VAE to fp32, and the stats span 1.5-3.3.
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model
