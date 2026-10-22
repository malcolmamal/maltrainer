from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.qwen_image_2_1.vae import LATENTS_MEAN, LATENTS_STD, QwenImage21VAE

from ._vae_helpers import randomize, stub_vae, use_stubs, use_tiny_networks


def _moments(mean: float, logvar: float) -> torch.Tensor:
    return torch.cat([torch.full((1, 64, 1, 1), mean), torch.full((1, 64, 1, 1), logvar)], dim=1)


def test_a_vae_without_a_decoder_has_neither_decoder_nor_post_quant_conv(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch, include_decoder=False)

    assert vae.decoder is None
    assert vae.post_quant_conv is None
    assert not any(key.startswith(("decoder.", "post_quant_conv.")) for key in vae.state_dict())


def test_the_state_dict_keys_start_with_the_four_checkpoint_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    use_stubs(monkeypatch)
    with torch.device("meta"):
        vae = QwenImage21VAE(include_decoder=True)

    assert {key.split(".")[0] for key in vae.state_dict()} == {"quant_conv", "post_quant_conv"}
    assert vae.quant_conv.weight.shape == (128, 128, 1, 1, 1)
    assert vae.post_quant_conv.weight.shape == (64, 64, 1, 1, 1)


def test_pixels_in_zero_to_one_are_mapped_to_minus_one_to_one_with_an_opaque_alpha_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vae = stub_vae(monkeypatch)
    x = torch.rand(2, 3, 16, 32, generator=torch.Generator().manual_seed(0))

    vae.encode(x)

    seen = vae.encoder.seen[0]
    assert seen.shape == (2, 4, 16, 32)
    torch.testing.assert_close(seen[:, :3], x * 2 - 1)
    assert torch.equal(seen[:, 3], torch.ones(2, 16, 32))


def test_pixels_that_dip_below_zero_are_taken_as_already_in_minus_one_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0)) * 2 - 1

    vae.encode(x)

    torch.testing.assert_close(vae.encoder.seen[0][:, :3], x)


def test_an_input_that_already_has_four_channels_is_not_padded(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    x = torch.rand(1, 4, 16, 16, generator=torch.Generator().manual_seed(0)) - 0.5

    vae.encode(x)

    assert torch.equal(vae.encoder.seen[0], x)


def test_a_deterministic_encode_returns_the_normalised_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    vae.encoder.moments = _moments(mean=1.0, logvar=3.0)

    latents = vae.encode(torch.zeros(2, 3, 16, 16))

    expected = (1.0 - torch.tensor(LATENTS_MEAN)) / torch.tensor(LATENTS_STD)
    assert latents.shape == (2, 64, 1, 1)
    torch.testing.assert_close(latents[0, :, 0, 0], expected)


def test_a_sampled_encode_adds_gaussian_noise_scaled_by_half_the_log_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch)
    vae.encoder.moments = _moments(mean=0.0, logvar=2.0)

    torch.manual_seed(5)
    latents = vae.encode(torch.zeros(1, 3, 16, 16), deterministic=False)

    torch.manual_seed(5)
    noise = torch.randn(1, 64, 1, 1)
    torch.testing.assert_close(latents, vae.normalize_latents(math.exp(1.0) * noise))


@pytest.mark.parametrize(("logvar", "std"), [(100.0, math.exp(10.0)), (-100.0, math.exp(-15.0))])
def test_the_log_variance_is_clamped_before_sampling(
    monkeypatch: pytest.MonkeyPatch, logvar: float, std: float
) -> None:
    vae = stub_vae(monkeypatch)
    vae.encoder.moments = _moments(mean=0.0, logvar=logvar)

    torch.manual_seed(1)
    latents = vae.encode(torch.zeros(1, 3, 16, 16), deterministic=False)

    torch.manual_seed(1)
    noise = torch.randn(1, 64, 1, 1)
    torch.testing.assert_close(vae.unnormalize_latents(latents), std * noise, rtol=1e-4, atol=1e-6)


def test_decoding_without_a_decoder_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = stub_vae(monkeypatch, include_decoder=False)

    with pytest.raises(RuntimeError, match="Decoder was not loaded"):
        vae.decode(torch.zeros(1, 64, 1, 1))


def test_decoding_unnormalises_takes_rgb_and_maps_minus_one_to_one_onto_zero_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vae = stub_vae(monkeypatch)
    target = torch.zeros(1, 64, 1, 2)
    target[0, 0], target[0, 1], target[0, 2], target[0, 3] = -2.0, 0.5, 3.0, 0.0
    target[0, 2, 0, 1] = -1.0

    image = vae.decode(vae.normalize_latents(target))

    assert image.shape == (1, 3, 1, 2)
    torch.testing.assert_close(image[0, :, 0, 0], torch.tensor([0.0, 0.75, 1.0]))
    torch.testing.assert_close(image[0, 2, 0, 1], torch.tensor(0.0))


def test_a_real_tiny_encoder_turns_an_image_into_sixteenth_resolution_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    use_tiny_networks(monkeypatch)
    vae = QwenImage21VAE(include_decoder=False)
    randomize(vae, std=0.1)
    x = torch.rand(2, 3, 32, 48, generator=torch.Generator().manual_seed(0))

    first = vae.encode(x)

    assert first.shape == (2, 64, 2, 3)
    torch.testing.assert_close(first, vae.encode(x), atol=0, rtol=0)
