from __future__ import annotations

import pytest

from maltrainer.models.minimax_h3.video_vae import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    VIDEO_CLIP_LENGTH,
    VIDEO_TOKEN_DROP,
    _map_encoder_key,
    align_frame_count,
    video_latent_frames,
)


def test_the_latent_statistics_cover_all_24_channels_with_positive_deviations() -> None:
    assert len(DEFAULT_LATENTS_MEAN) == len(DEFAULT_LATENTS_STD) == 24
    assert min(DEFAULT_LATENTS_STD) > 0
    assert DEFAULT_LATENTS_MEAN[0] == pytest.approx(0.858090341091156)
    assert DEFAULT_LATENTS_STD[-1] == pytest.approx(2.6127843856811523)


def test_chunking_constants_match_the_reference_vae() -> None:
    assert (VIDEO_CLIP_LENGTH, VIDEO_TOKEN_DROP) == (17, 3)


@pytest.mark.parametrize(("frames", "expected"), [(5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (100, 90)])
def test_a_frame_count_is_aligned_down_to_the_17k_plus_5_grid(frames: int, expected: int) -> None:
    assert align_frame_count(frames) == expected


@pytest.mark.parametrize("frames", [0, 4])
def test_clips_shorter_than_five_frames_cannot_be_aligned(frames: int) -> None:
    with pytest.raises(ValueError, match="at least 5 frames"):
        align_frame_count(frames)


@pytest.mark.parametrize(("frames", "latents"), [(1, 1), (5, 2), (22, 7), (39, 12), (56, 17)])
def test_a_17k_plus_5_clip_has_5k_plus_2_latent_frames(frames: int, latents: int) -> None:
    assert video_latent_frames(frames) == latents


@pytest.mark.parametrize(
    ("key", "mapped"),
    [
        ("encoder.conv_in.weight", "conv_in.weight"),
        ("encoder.norm_out.bias", "norm_out.bias"),
        ("quant_conv.weight", "quant_conv.weight"),
        ("encoder.down.2.block.1.conv1.weight", "down_blocks.2.resnets.1.conv1.weight"),
        ("encoder.down.0.block.0.nin_shortcut.weight", "down_blocks.0.resnets.0.conv_shortcut.weight"),
        ("encoder.down.3.downsample.conv.bias", "down_blocks.3.downsamplers.0.conv.bias"),
        ("encoder.mid.block.1.nin_shortcut.weight", "mid_block.resnets.1.conv_shortcut.weight"),
        ("decoder.conv_in.weight", None),
        ("post_quant_conv.weight", None),
    ],
)
def test_checkpoint_keys_are_renamed_to_the_module_layout_and_decoder_keys_dropped(
    key: str, mapped: str | None
) -> None:
    assert _map_encoder_key(key) == mapped
