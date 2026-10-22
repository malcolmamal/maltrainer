from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.minimax_h3.audio_vae import DEFAULT_LATENTS_MEAN, DEFAULT_LATENTS_STD, MiniMaxH3AudioEncoder
from tests.maltrainer.models.minimax_h3.h3_audio import tiny_audio_encoder, wave


def test_the_latent_statistics_cover_all_32_channels() -> None:
    assert len(DEFAULT_LATENTS_MEAN) == len(DEFAULT_LATENTS_STD) == 32
    assert min(DEFAULT_LATENTS_STD) > 1.0
    assert DEFAULT_LATENTS_MEAN[1] == pytest.approx(0.3876466479950502)


@pytest.mark.parametrize(("length", "frames"), [(1, 1), (800, 1), (801, 2), (1600, 2), (4000, 5)])
def test_a_stereo_waveform_gives_32_channels_by_two_by_ceil_of_length_over_800(
    monkeypatch: pytest.MonkeyPatch, length: int, frames: int
) -> None:
    out = tiny_audio_encoder(monkeypatch).encode(wave(length, batch=2))

    assert out.shape == (2, 32, 2, frames)


def test_each_stereo_channel_goes_through_the_mono_encoder_as_its_own_batch_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = tiny_audio_encoder(monkeypatch)
    stereo = wave(1600)

    both = encoder.encode(stereo)

    torch.testing.assert_close(both[:, :, 0], encoder.encode(stereo[:, :1])[:, :, 0])
    torch.testing.assert_close(both[:, :, 1], encoder.encode(stereo[:, 1:])[:, :, 0])


def test_a_ragged_tail_is_zero_padded_to_a_whole_latent_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = tiny_audio_encoder(monkeypatch)
    ragged = wave(1000)

    torch.testing.assert_close(encoder.encode(ragged), encoder.encode(F.pad(ragged, (0, 600))))


def test_latents_are_normalised_with_the_mean_and_std_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = tiny_audio_encoder(monkeypatch)
    audio = wave(800)
    plain = encoder.encode(audio)

    encoder.latents_mean.fill_(1.0)
    encoder.latents_std.fill_(2.0)
    shifted = encoder.encode(audio)

    expected = (
        plain * torch.tensor(DEFAULT_LATENTS_STD).view(1, -1, 1, 1)
        + torch.tensor(DEFAULT_LATENTS_MEAN).view(1, -1, 1, 1)
        - 1.0
    ) / 2.0
    torch.testing.assert_close(shifted, expected, atol=1e-5, rtol=1e-5)


def test_encoding_is_deterministic_float32_and_tracks_no_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = tiny_audio_encoder(monkeypatch)
    audio = wave(800).half()

    first, second = encoder.encode(audio), encoder.encode(audio)

    assert torch.equal(first, second)
    assert first.dtype == torch.float32
    assert not first.requires_grad


def test_the_released_configuration_encodes_one_second_to_forty_latent_frames() -> None:
    with torch.device("meta"):
        out = MiniMaxH3AudioEncoder().encode(torch.empty(1, 2, 32000))

    assert out.shape == (1, 32, 2, 40)
