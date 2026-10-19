from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.vae import Resample2D, Resample3D, VAEAttention

from ._vae_helpers import randomize


def _box_kernel(conv: torch.nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()


def _center_kernel(conv: torch.nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0
        conv.bias.zero_()


@pytest.mark.parametrize("layer", [Resample2D, Resample3D])
def test_downsampling_pads_the_bottom_and_right_then_strides_a_3x3_conv_by_two(layer: type) -> None:
    resample = layer(1, "down")
    _box_kernel(resample.resample[1])
    x = torch.arange(16.0).view(1, 1, 4, 4)

    out = resample(x)

    assert out.shape == (1, 1, 2, 2)
    assert out[0, 0, 0, 0] == x[0, 0, :3, :3].sum()
    assert out[0, 0, 1, 1] == x[0, 0, 2:, 2:].sum()


@pytest.mark.parametrize("layer", [Resample2D, Resample3D])
def test_downsampling_an_odd_size_rounds_down(layer: type) -> None:
    out = randomize(layer(2, "down"))(torch.zeros(1, 2, 5, 7))

    assert out.shape == (1, 2, 2, 3)


@pytest.mark.parametrize("layer", [Resample2D, Resample3D])
def test_upsampling_repeats_each_pixel_into_a_2x2_block_before_the_conv(layer: type) -> None:
    resample = layer(1, "up")
    _center_kernel(resample.resample[1])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(1, 1, 2, 2)

    out = resample(x)

    assert out.shape == (1, 1, 4, 4)
    torch.testing.assert_close(out, x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


def test_the_3d_resamplers_carry_a_temporal_conv_for_the_checkpoint_that_single_frames_never_use() -> None:
    down, up = Resample3D(3, "down"), Resample3D(3, "up")
    x = torch.randn(1, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    before = (randomize(down)(x), randomize(up)(x))

    for module in (down, up):
        torch.nn.init.constant_(module.time_conv.weight, 7.0)

    assert down.time_conv.weight.shape == (3, 3, 1, 1, 1)
    assert up.time_conv.weight.shape == (6, 3, 1, 1, 1)
    torch.testing.assert_close((down(x), up(x)), before)


def test_the_2d_resampler_has_no_temporal_conv() -> None:
    assert sorted(Resample2D(2, "down").state_dict()) == ["resample.1.bias", "resample.1.weight"]


def test_attention_over_a_single_position_returns_the_value_so_the_output_adds_the_normed_input() -> None:
    attn = VAEAttention(2)
    with torch.no_grad():
        attn.to_qkv.weight.zero_()
        attn.to_qkv.bias.zero_()
        attn.to_qkv.weight[4:, :, 0, 0] = torch.eye(2)
        attn.proj.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
        attn.proj.bias.zero_()
    x = torch.tensor([3.0, 4.0]).view(1, 2, 1, 1)

    out = attn(x)

    torch.testing.assert_close(out, x + attn.norm(x))


def test_zero_queries_and_keys_make_each_pixel_receive_the_mean_value_of_all_pixels() -> None:
    attn = VAEAttention(2)
    with torch.no_grad():
        attn.to_qkv.weight.zero_()
        attn.to_qkv.bias.zero_()
        attn.to_qkv.weight[4:, :, 0, 0] = torch.eye(2)
        attn.proj.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
        attn.proj.bias.zero_()
    x = torch.randn(1, 2, 2, 3, generator=torch.Generator().manual_seed(0))

    out = attn(x)

    mean_value = attn.norm(x).mean(dim=(2, 3), keepdim=True)
    torch.testing.assert_close(out, x + mean_value.expand_as(x))


def test_attention_with_a_zero_projection_is_the_identity_and_keeps_the_shape() -> None:
    attn = randomize(VAEAttention(4))
    with torch.no_grad():
        attn.proj.weight.zero_()
        attn.proj.bias.zero_()
    x = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(attn(x), x)
