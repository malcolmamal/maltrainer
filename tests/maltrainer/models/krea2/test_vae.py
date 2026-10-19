from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.krea2.vae import (
    LATENTS_MEAN,
    LATENTS_STD,
    AttentionBlock,
    CausalConv2d,
    Krea2VAE,
    Resample,
    ResidualBlock,
    RMSNorm,
    WanImageEncoder,
)


def _filled(module: nn.Module, seed: int = 0) -> nn.Module:
    """CausalConv2d allocates its weight with torch.empty, so every test fills parameters itself."""
    generator = torch.Generator().manual_seed(seed)
    for parameter in module.parameters():
        parameter.data = torch.randn(parameter.shape, generator=generator) * 0.2
    return module.eval()


class Fixed(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x
        return self.output


def test_a_causal_conv_on_one_frame_uses_only_the_last_temporal_tap_of_its_kernel() -> None:
    conv = CausalConv2d(2, 3, 3, padding=1)
    _filled(conv)
    x = torch.randn(1, 2, 5, 5)
    expected = torch.nn.functional.conv2d(x, conv.weight[:, :, 2], conv.bias, padding=1)
    conv.weight.data[:, :, :2] = 1e3

    assert conv.weight.shape == (3, 2, 3, 3, 3)
    torch.testing.assert_close(conv(x), expected)


def test_the_wan_rms_norm_normalises_over_channels_and_scales_by_root_c_and_gamma() -> None:
    norm = RMSNorm(2, images=False)
    norm.gamma.data = torch.tensor([2.0, 1.0]).view(2, 1, 1, 1)

    out = norm(torch.tensor([3.0, 4.0]).view(1, 2, 1, 1))

    torch.testing.assert_close(out.flatten(), torch.tensor([0.6 * 2**0.5 * 2.0, 0.8 * 2**0.5]))
    assert RMSNorm(2, images=True).gamma.shape == (2, 1, 1)


def test_a_residual_block_adds_a_shortcut_convolution_only_when_the_width_changes() -> None:
    same, wider = ResidualBlock(4, 4), ResidualBlock(4, 8)
    _filled(wider)
    x = torch.randn(1, 4, 6, 6)

    assert isinstance(same.shortcut, nn.Identity)
    assert isinstance(wider.shortcut, CausalConv2d)
    assert wider(x).shape == (1, 8, 6, 6)


def test_an_attention_block_with_a_closed_projection_is_the_identity() -> None:
    block = _filled(AttentionBlock(4))
    block.proj.weight.data.zero_()
    block.proj.bias.data.zero_()
    x = torch.randn(2, 4, 3, 3)

    assert torch.equal(block(x), x)


def test_resampling_pads_right_and_bottom_and_keeps_every_second_pixel_for_both_variants() -> None:
    for temporal in (False, True):
        resample = Resample(1, temporal)
        resample.resample[1].weight.data.zero_()
        resample.resample[1].weight.data[0, 0, 0, 0] = 1.0
        resample.resample[1].bias.data.zero_()

        out = resample(torch.arange(16.0).view(1, 1, 4, 4))

        assert out[0, 0].tolist() == [[0, 2], [8, 10]]


def test_the_encoder_downsamples_eight_times_to_twice_the_latent_channels() -> None:
    encoder = _filled(WanImageEncoder(dim=4))

    assert encoder(torch.randn(1, 3, 16, 24)).shape == (1, 32, 2, 3)


def test_pixels_are_mapped_to_minus_one_one_and_the_posterior_mean_is_normalised_per_channel() -> None:
    model = _filled(Krea2VAE(dim=4))
    moments = torch.randn(1, 32, 2, 3, generator=torch.Generator().manual_seed(1))
    model.encoder = Fixed(moments)
    model.conv1.weight.data = torch.eye(32).view(32, 32, 1, 1, 1)
    model.conv1.bias.data.zero_()

    latents = model.encode(torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 16, 24))

    assert model.encoder.seen[0, :, 0, 0].tolist() == [-1.0, 0.0, 1.0]
    mean, std = torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1), torch.tensor(LATENTS_STD).view(1, -1, 1, 1)
    torch.testing.assert_close(latents, (moments[:, :16] - mean) / std)


def test_a_bf16_model_returns_float32_latents() -> None:
    model = _filled(Krea2VAE(dim=4)).to(torch.bfloat16)

    latents = model.encode(torch.rand(1, 3, 16, 16))

    assert latents.dtype == torch.float32 and latents.shape == (1, 16, 2, 2)
    assert not latents.requires_grad


def test_the_wan21_statistics_are_not_part_of_the_state_dict() -> None:
    keys = set(Krea2VAE(dim=4).state_dict())

    assert "latents_mean" not in keys and "latents_std" not in keys
    assert {"conv1.weight", "encoder.conv1.weight", "encoder.head.2.bias"} <= keys


@pytest.mark.parametrize("stats", [LATENTS_MEAN, LATENTS_STD])
def test_the_statistics_cover_all_sixteen_latent_channels(stats: tuple) -> None:
    assert len(stats) == 16
