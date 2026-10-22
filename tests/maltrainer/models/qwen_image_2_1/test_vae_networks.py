from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.vae import Qwen21VAEDecoder, Qwen21VAEEncoder

from ._vae_helpers import randomize


@pytest.fixture(scope="module")
def full_encoder() -> Qwen21VAEEncoder:
    with torch.device("meta"):
        return Qwen21VAEEncoder()


@pytest.fixture(scope="module")
def full_decoder() -> Qwen21VAEDecoder:
    with torch.device("meta"):
        return Qwen21VAEDecoder()


def _tiny_encoder() -> Qwen21VAEEncoder:
    return randomize(Qwen21VAEEncoder(base_dim=4, z_dim=2), std=0.3)


def test_the_encoder_downsamples_sixteen_fold_and_emits_mean_and_logvar_channels() -> None:
    out = _tiny_encoder()(torch.randn(2, 4, 32, 48, generator=torch.Generator().manual_seed(0)))

    assert out.shape == (2, 4, 2, 3)
    assert torch.isfinite(out).all()


def test_the_encoder_is_deterministic_and_responds_to_its_input() -> None:
    encoder = _tiny_encoder()
    x = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(0))

    first, second, other = encoder(x), encoder(x), encoder(x + 1.0)

    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.allclose(first, other)


def test_the_encoder_widens_across_stages_and_the_last_stage_keeps_the_resolution(
    full_encoder: Qwen21VAEEncoder,
) -> None:
    stages = full_encoder.downsamples

    widths = [stage.downsamples[0].residual[2].weight.shape[0] for stage in stages]
    assert widths == [96, 192, 384, 768, 768]
    assert [len(stage.downsamples) for stage in stages] == [3, 3, 3, 3, 2]


def test_the_encoder_head_emits_twice_the_latent_channels(full_encoder: Qwen21VAEEncoder) -> None:
    assert full_encoder.conv1.weight.shape == (96, 4, 1, 3, 3)
    assert full_encoder.head[2].weight.shape == (128, 768, 1, 3, 3)
    assert [type(m).__name__ for m in full_encoder.middle] == ["ResBlock", "VAEAttention", "ResBlock"]


def test_the_full_size_encoder_maps_pixels_to_a_sixteenth_resolution_grid(full_encoder: Qwen21VAEEncoder) -> None:
    with torch.device("meta"):
        out = full_encoder(torch.zeros(1, 4, 64, 32))

    assert out.shape == (1, 128, 4, 2)


def test_the_decoder_upsamples_sixteen_fold_to_four_channels(full_decoder: Qwen21VAEDecoder) -> None:
    with torch.device("meta"):
        out = full_decoder(torch.zeros(1, 64, 2, 3))

    assert out.shape == (1, 4, 32, 48)


def test_the_decoder_narrows_across_stages_and_only_the_first_four_resample(full_decoder: Qwen21VAEDecoder) -> None:
    stages = full_decoder.upsamples

    assert [len(stage.upsamples) for stage in stages] == [4, 4, 4, 4, 3]
    assert [stage.upsamples[2].residual[6].weight.shape[0] for stage in stages] == [1152, 1152, 576, 288, 144]
    assert stages[4].avg_shortcut is None
    assert full_decoder.head[2].weight.shape == (4, 144, 1, 3, 3)


def test_the_decoder_keys_name_stage_blocks_resamplers_and_the_stem(full_decoder: Qwen21VAEDecoder) -> None:
    keys = set(full_decoder.state_dict())

    assert {"conv1.weight", "middle.1.to_qkv.weight", "head.2.bias"} <= keys
    assert "upsamples.0.upsamples.3.time_conv.weight" in keys
    assert "upsamples.3.upsamples.3.resample.1.weight" in keys
    assert "upsamples.3.upsamples.3.time_conv.weight" not in keys
    assert not any("avg_shortcut" in key for key in keys)


def test_the_encoder_keys_name_stage_blocks_resamplers_and_the_stem(full_encoder: Qwen21VAEEncoder) -> None:
    keys = set(full_encoder.state_dict())

    assert {"conv1.weight", "middle.1.proj.weight", "head.0.gamma"} <= keys
    assert "downsamples.0.downsamples.0.residual.0.gamma" in keys
    assert "downsamples.1.downsamples.2.time_conv.weight" in keys
    assert "downsamples.0.downsamples.2.time_conv.weight" not in keys
    assert not any("avg_shortcut" in key for key in keys)
