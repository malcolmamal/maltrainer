from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3 import audio_vae

STRIDES = (2, 4, 4, 5, 5)


def use_tiny_audio_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the hard-coded 2048-wide DAC encoder and 2048 -> 32 projection; strides and the 32 latents stay."""
    dac, projection = audio_vae.DacEncoder, audio_vae.AttnProjection
    monkeypatch.setattr(audio_vae, "DacEncoder", lambda _d, strides, _latent: dac(2, strides, 8))
    monkeypatch.setattr(audio_vae, "AttnProjection", lambda _i, out, **_options: projection(8, out, num_heads=2))


def tiny_audio_encoder(monkeypatch: pytest.MonkeyPatch, seed: int = 0) -> audio_vae.MiniMaxH3AudioEncoder:
    use_tiny_audio_encoder(monkeypatch)
    torch.manual_seed(seed)
    return audio_vae.MiniMaxH3AudioEncoder().eval()


def wave(length: int, batch: int = 1, channels: int = 2, seed: int = 0) -> torch.Tensor:
    return torch.rand(batch, channels, length, generator=torch.Generator().manual_seed(seed)) * 2 - 1
