from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.vae import LATENTS_MEAN, LATENTS_STD

from ._vae_helpers import stub_vae

MEAN = torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1)
STD = torch.tensor(LATENTS_STD).view(1, -1, 1, 1)


def _moments(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.cat([mean, logvar], dim=1)


def test_unit_range_pixels_are_mapped_to_minus_one_one_before_the_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    pixels = torch.tensor([0.0, 0.25, 1.0]).view(1, 1, 1, 3).expand(1, 3, 1, 3)

    vae.encode(pixels)

    seen = vae.encoder.seen[0]
    assert seen[0, 0, 0].tolist() == [-1.0, -0.5, 1.0]


def test_pixels_already_in_minus_one_one_are_passed_through_unscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    pixels = torch.tensor([-1.0, 0.0, 0.5]).view(1, 1, 1, 3).expand(1, 3, 1, 3)

    vae.encode(pixels)

    assert vae.encoder.seen[0][0, 0, 0].tolist() == [-1.0, 0.0, 0.5]


def test_rgb_input_gets_an_opaque_alpha_channel_but_rgba_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    rgb = torch.full((2, 3, 2, 2), -0.5)
    rgba = torch.full((2, 4, 2, 2), -0.5)

    vae.encode(rgb)
    vae.encode(rgba)

    padded, untouched = vae.encoder.seen
    assert padded.shape == (2, 4, 2, 2) and torch.equal(padded[:, 3], torch.ones(2, 2, 2))
    assert torch.equal(padded[:, :3], rgb)
    assert torch.equal(untouched, rgba)


def test_deterministic_encoding_returns_the_normalised_posterior_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    mean = torch.linspace(-2.0, 3.0, 64).view(1, 64, 1, 1)
    vae.encoder.moments = _moments(mean, torch.full((1, 64, 1, 1), 5.0))

    latents = vae.encode(torch.rand(2, 3, 4, 4))

    assert latents.shape == (2, 64, 1, 1)
    assert torch.allclose(latents[0], ((mean - MEAN) / STD)[0], atol=1e-6)


def test_sampling_adds_noise_scaled_by_half_the_log_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    mean, logvar = torch.zeros(1, 64, 1, 1), torch.full((1, 64, 1, 1), 2.0)
    vae.encoder.moments = _moments(mean, logvar)
    pixels = torch.rand(1, 3, 4, 4)

    torch.manual_seed(7)
    sampled = vae.encode(pixels, deterministic=False)
    torch.manual_seed(7)
    noise = torch.randn(1, 64, 1, 1)

    expected = (mean + torch.exp(torch.tensor(1.0)) * noise - MEAN) / STD
    assert torch.allclose(sampled, expected, atol=1e-5)


def test_the_log_variance_is_clamped_so_extremes_cannot_overflow_or_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    huge, tiny = torch.full((1, 64, 1, 1), 1000.0), torch.full((1, 64, 1, 1), -1000.0)
    mean, pixels = torch.zeros(1, 64, 1, 1), torch.rand(1, 3, 4, 4)

    vae.encoder.moments = _moments(mean, huge)
    torch.manual_seed(1)
    wide = vae.encode(pixels, deterministic=False)
    vae.encoder.moments = _moments(mean, tiny)
    torch.manual_seed(1)
    narrow = vae.encode(pixels, deterministic=False)

    noise = torch.randn(1, 64, 1, 1, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(wide).all() and torch.allclose(narrow, (mean - MEAN) / STD, atol=1e-5)
    assert torch.allclose(wide, (torch.exp(torch.tensor(10.0)) * noise - MEAN) / STD, rtol=1e-4)


def test_decoding_without_a_decoder_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch, include_decoder=False)

    with pytest.raises(RuntimeError, match="Decoder was not loaded"):
        vae.decode(torch.zeros(1, 64, 1, 1))


def test_decoding_unnormalises_the_latents_before_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    z = torch.randn(2, 64, 1, 1)

    vae.decode(z)

    assert torch.allclose(vae.decoder.seen[0], z * STD + MEAN, atol=1e-5)


def test_decoding_keeps_rgb_rescales_minus_one_one_to_unit_range_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    # the stub decoder returns its first four latent channels, so choose latents that unnormalise to known values
    wanted = torch.zeros(1, 64, 1, 1)
    wanted[0, :4, 0, 0] = torch.tensor([-1.0, 0.0, 3.0, 9.0])
    z = (wanted - MEAN) / STD

    image = vae.decode(z)

    assert image.shape == (1, 3, 1, 1)
    assert torch.allclose(image.flatten(), torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)
