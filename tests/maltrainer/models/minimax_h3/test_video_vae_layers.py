from __future__ import annotations

import torch

from maltrainer.models.minimax_h3.video_vae import (
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoDownBlock3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoMidBlock3d,
    MiniMaxH3VideoResnetBlock3d,
)


def _x(frames: int = 5, channels: int = 4, size: int = 8, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, channels, frames, size, size, generator=torch.Generator().manual_seed(seed))


def test_a_causal_conv_keeps_the_frame_count_and_the_spatial_size() -> None:
    conv = MiniMaxH3VideoCausalConv3d(4, 6, 3, spatial_padding=1, temporal_padding=2)

    assert conv(_x()).shape == (1, 6, 5, 8, 8)


def test_a_causal_conv_output_never_depends_on_later_frames() -> None:
    torch.manual_seed(0)
    conv = MiniMaxH3VideoCausalConv3d(4, 4, 3, spatial_padding=1, temporal_padding=2)
    x = _x()
    changed = x.clone()
    changed[:, :, 3:] += 5.0

    torch.testing.assert_close(conv(x)[:, :, :3], conv(changed)[:, :, :3])
    assert not torch.allclose(conv(x)[:, :, 3:], conv(changed)[:, :, 3:])


def test_a_causal_conv_pads_the_start_with_zeros_in_time() -> None:
    conv = MiniMaxH3VideoCausalConv3d(1, 1, (3, 1, 1), temporal_padding=2)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()

    out = conv(torch.arange(1.0, 5.0).view(1, 1, 4, 1, 1))

    assert out.flatten().tolist() == [1.0, 3.0, 6.0, 9.0]


def test_a_stride_shrinks_the_output_without_a_padding_option() -> None:
    conv = MiniMaxH3VideoCausalConv3d(4, 4, 3, stride=(2, 2, 2), temporal_padding=2)

    assert conv(_x(frames=5, size=9)).shape == (1, 4, 3, 4, 4)


def test_group_norm_normalises_every_frame_on_its_own() -> None:
    norm = MiniMaxH3VideoGroupNorm(2, 4)
    x = _x(frames=2)
    x[:, :, 1] = x[:, :, 0] * 5.0 + 3.0

    out = norm(x)

    torch.testing.assert_close(out[:, :, 0], out[:, :, 1], atol=1e-4, rtol=1e-4)
    assert out.shape == x.shape


def test_a_resnet_block_has_a_shortcut_only_when_the_width_changes() -> None:
    same = MiniMaxH3VideoResnetBlock3d(4, 4, norm_num_groups=2)
    wider = MiniMaxH3VideoResnetBlock3d(4, 8, norm_num_groups=2)

    assert same.conv_shortcut is None
    assert wider.conv_shortcut is not None
    assert wider(_x()).shape == (1, 8, 5, 8, 8)


def test_a_resnet_block_adds_its_input_when_the_branch_is_zero() -> None:
    block = MiniMaxH3VideoResnetBlock3d(4, 4, norm_num_groups=2)
    torch.nn.init.zeros_(block.conv2.weight)
    torch.nn.init.zeros_(block.conv2.bias)
    x = _x()

    torch.testing.assert_close(block(x), x)


def test_a_downsample_by_two_halves_height_and_width_and_the_frames() -> None:
    down = MiniMaxH3VideoDownsample3d(4, 4, temporal_stride=2, spatial_stride=2)

    assert down(_x(frames=9, size=16)).shape == (1, 4, 5, 8, 8)


def test_a_down_block_downsamples_only_when_a_factor_is_above_one() -> None:
    plain = MiniMaxH3VideoDownBlock3d(4, 4, 1, 1, 1, norm_num_groups=2)
    shrinking = MiniMaxH3VideoDownBlock3d(4, 8, 2, 2, 2, norm_num_groups=2)

    assert plain.downsamplers is None
    assert len(shrinking.resnets) == 2
    assert plain(_x()).shape == (1, 4, 5, 8, 8)
    assert shrinking(_x(frames=5)).shape == (1, 8, 3, 4, 4)


def test_a_mid_block_is_two_width_preserving_resnets() -> None:
    mid = MiniMaxH3VideoMidBlock3d(4, norm_num_groups=2)

    assert len(mid.resnets) == 2
    assert mid(_x()).shape == (1, 4, 5, 8, 8)


def test_a_temporal_only_downsample_halves_the_frames_without_spatial_padding() -> None:
    down = MiniMaxH3VideoDownsample3d(4, 4, temporal_stride=2, spatial_stride=1)

    assert down(_x(frames=9, size=8)).shape[2] == 5
