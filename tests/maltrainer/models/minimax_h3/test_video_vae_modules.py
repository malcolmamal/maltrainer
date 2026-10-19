from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.minimax_h3.video_vae import (
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoDownBlock3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoMidBlock3d,
    MiniMaxH3VideoResnetBlock3d,
)


def _random(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn_like(p) * 0.3
    return module.eval()


def test_a_causal_conv_pads_space_by_reflection_and_time_with_zeros_on_the_past_side_only() -> None:
    conv = _random(MiniMaxH3VideoCausalConv3d(2, 3, 3, spatial_padding=1, temporal_padding=2))
    x = torch.randn(1, 2, 4, 5, 6)

    padded = F.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
    padded = F.pad(padded, (0, 0, 0, 0, 2, 0))
    expected = F.conv3d(padded, conv.weight, conv.bias)

    out = conv(x)
    assert out.shape == (1, 3, 4, 5, 6)
    torch.testing.assert_close(out, expected)


def test_a_causal_conv_never_lets_a_frame_see_later_frames() -> None:
    conv = _random(MiniMaxH3VideoCausalConv3d(2, 2, 3, spatial_padding=1, temporal_padding=2))
    x = torch.randn(1, 2, 6, 4, 4)
    changed = x.clone()
    changed[:, :, 3:] += 4.0

    out, out_changed = conv(x), conv(changed)

    torch.testing.assert_close(out[:, :, :3], out_changed[:, :, :3])
    assert not torch.allclose(out[:, :, 3:], out_changed[:, :, 3:])


def test_a_causal_conv_without_padding_is_a_plain_valid_convolution() -> None:
    conv = _random(MiniMaxH3VideoCausalConv3d(2, 2, 1))
    x = torch.randn(1, 2, 3, 4, 4)

    torch.testing.assert_close(conv(x), F.conv3d(x, conv.weight, conv.bias))


def test_group_norm_normalises_each_frame_on_its_own() -> None:
    norm = _random(MiniMaxH3VideoGroupNorm(2, 4))
    x = torch.randn(2, 4, 3, 5, 5) * 3 + 2

    frames = x.permute(0, 2, 1, 3, 4).reshape(6, 4, 5, 5)
    expected = F.group_norm(frames, 2, norm.weight, norm.bias, norm.eps).reshape(2, 3, 4, 5, 5).permute(0, 2, 1, 3, 4)

    out = norm(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected)


def test_group_norm_of_one_frame_is_unaffected_by_the_other_frames() -> None:
    norm = MiniMaxH3VideoGroupNorm(2, 4)
    x = torch.randn(1, 4, 3, 5, 5)
    changed = x.clone()
    changed[:, :, 1:] *= 10

    torch.testing.assert_close(norm(x)[:, :, 0], norm(changed)[:, :, 0])


def test_a_resnet_block_adds_its_input_when_channels_match_and_a_1x1_shortcut_when_they_differ() -> None:
    same = _random(MiniMaxH3VideoResnetBlock3d(4, 4, norm_num_groups=2))
    widening = _random(MiniMaxH3VideoResnetBlock3d(4, 8, norm_num_groups=2))
    x = torch.randn(1, 4, 3, 6, 6)

    assert same.conv_shortcut is None and widening.conv_shortcut is not None
    body = same.conv2(F.silu(same.norm2(same.conv1(F.silu(same.norm1(x))))))
    torch.testing.assert_close(same(x), x + body)
    assert widening(x).shape == (1, 8, 3, 6, 6)
    torch.testing.assert_close(
        widening(x) - widening.conv_shortcut(x),
        widening.conv2(F.silu(widening.norm2(widening.conv1(F.silu(widening.norm1(x)))))),
    )


@pytest.mark.parametrize(
    "temporal, spatial, shape",
    [(1, 2, (1, 4, 5, 4, 4)), (2, 2, (1, 4, 3, 4, 4)), (2, 1, (1, 4, 3, 6, 6)), (1, 1, (1, 4, 5, 6, 6))],
)
# spatial stride 1 applies the unpadded 3x3 kernel, so the image shrinks by 2 px (no shipped config does this)
def test_a_downsampler_strides_time_and_space_by_the_given_factors(temporal: int, spatial: int, shape: tuple) -> None:
    down = _random(MiniMaxH3VideoDownsample3d(4, 4, temporal_stride=temporal, spatial_stride=spatial))

    out = down(torch.randn(1, 4, 5, 8, 8))

    assert out.shape == shape


def test_a_down_block_only_gets_a_downsampler_when_it_changes_the_resolution() -> None:
    plain = MiniMaxH3VideoDownBlock3d(4, 8, 2, 1, 1, norm_num_groups=2)
    spatial = MiniMaxH3VideoDownBlock3d(4, 8, 2, 1, 2, norm_num_groups=2)
    temporal = MiniMaxH3VideoDownBlock3d(4, 8, 2, 2, 1, norm_num_groups=2)

    assert plain.downsamplers is None
    assert len(spatial.downsamplers) == 1 and len(temporal.downsamplers) == 1
    assert [r.in_channels for r in plain.resnets] == [4, 8]  # only the first resnet widens


def test_a_down_block_runs_its_resnets_then_its_downsampler() -> None:
    block = _random(MiniMaxH3VideoDownBlock3d(4, 8, 2, 1, 2, norm_num_groups=2))
    x = torch.randn(1, 4, 3, 8, 8)

    expected = block.downsamplers[0](block.resnets[1](block.resnets[0](x)))

    out = block(x)
    assert out.shape == (1, 8, 3, 4, 4)
    torch.testing.assert_close(out, expected)


def test_the_mid_block_is_two_resnets_at_constant_width() -> None:
    mid = _random(MiniMaxH3VideoMidBlock3d(4, norm_num_groups=2))
    x = torch.randn(1, 4, 2, 4, 4)

    assert len(mid.resnets) == 2
    torch.testing.assert_close(mid(x), mid.resnets[1](mid.resnets[0](x)))
