from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.qwen_image_2_1.vae import CausalConv3d, ResBlock, RMSNorm3d, RMSNormAttn

from ._vae_helpers import randomize


def test_the_conv_stores_a_single_frame_kernel_and_applies_it_as_a_2d_convolution() -> None:
    conv = randomize(CausalConv3d(3, 5, kernel_size=3, padding=1))
    x = torch.randn(2, 3, 6, 4, generator=torch.Generator().manual_seed(0))

    out = conv(x)

    assert conv.weight.shape == (5, 3, 1, 3, 3)
    assert conv.bias.shape == (5,)
    torch.testing.assert_close(out, F.conv2d(x, conv.weight[:, :, 0], conv.bias, padding=1))


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "expected"),
    [(3, 1, 1, (6, 4)), (3, 1, 0, (4, 2)), (1, 1, 0, (6, 4)), (3, 2, 1, (3, 2))],
)
def test_the_conv_output_size_follows_kernel_stride_and_padding(
    kernel: int, stride: int, padding: int, expected: tuple[int, int]
) -> None:
    conv = randomize(CausalConv3d(1, 1, kernel, stride=stride, padding=padding))

    out = conv(torch.zeros(1, 1, 6, 4))

    assert out.shape[2:] == expected


def test_a_pointwise_conv_mixes_channels_with_its_weight_and_adds_the_bias() -> None:
    conv = CausalConv3d(2, 1, 1)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([2.0, -1.0]).view(1, 2, 1, 1, 1))
        conv.bias.fill_(0.5)

    out = conv(torch.tensor([3.0, 4.0]).view(1, 2, 1, 1))

    torch.testing.assert_close(out, torch.tensor([[[[2.5]]]]))


def test_the_channel_norm_scales_each_pixel_vector_to_root_channels_times_gamma() -> None:
    norm = RMSNorm3d(2)
    with torch.no_grad():
        norm.gamma.copy_(torch.tensor([1.0, 2.0]).view(2, 1, 1, 1))

    out = norm(torch.tensor([3.0, 4.0]).view(1, 2, 1, 1))

    torch.testing.assert_close(out.flatten(), torch.tensor([0.6, 1.6]) * math.sqrt(2.0))
    assert norm.gamma.shape == (2, 1, 1, 1)


def test_the_attention_norm_has_the_same_maths_with_a_three_dimensional_gamma() -> None:
    plain, attn = RMSNorm3d(3), RMSNormAttn(3)
    x = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0))

    assert attn.gamma.shape == (3, 1, 1)
    torch.testing.assert_close(attn(x), plain(x))


def test_the_channel_norms_keep_the_input_dtype() -> None:
    x = torch.randn(1, 4, 2, 2).to(torch.bfloat16)

    assert RMSNorm3d(4).to(torch.bfloat16)(x).dtype == torch.bfloat16
    assert RMSNormAttn(4).to(torch.bfloat16)(x).dtype == torch.bfloat16


def test_a_block_between_equal_widths_has_an_identity_shortcut_and_adds_its_residual() -> None:
    block = randomize(ResBlock(4, 4))
    x = torch.randn(1, 4, 3, 3, generator=torch.Generator().manual_seed(0))

    out = block(x)

    assert isinstance(block.shortcut, torch.nn.Identity)
    torch.testing.assert_close(out, x + block.residual(x))


def test_a_block_that_changes_width_projects_its_shortcut_with_a_pointwise_conv() -> None:
    block = randomize(ResBlock(2, 6))
    x = torch.randn(1, 2, 3, 3, generator=torch.Generator().manual_seed(0))

    out = block(x)

    assert block.shortcut.weight.shape == (6, 2, 1, 1, 1)
    assert out.shape == (1, 6, 3, 3)
    torch.testing.assert_close(out, block.shortcut(x) + block.residual(x))


def test_a_block_with_zeroed_output_convolution_passes_its_input_through() -> None:
    block = randomize(ResBlock(4, 4))
    with torch.no_grad():
        block.residual[6].weight.zero_()
        block.residual[6].bias.zero_()
    x = torch.randn(1, 4, 3, 3, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(block(x), x)
