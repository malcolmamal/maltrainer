from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3.video_vae import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    VIDEO_CLIP_LENGTH,
    MiniMaxH3VideoEncoderModel,
    align_frame_count,
    video_latent_frames,
)

LATENTS = 24


def small_encoder(seed: int = 0) -> MiniMaxH3VideoEncoderModel:
    """The real topology (/16 space, /4 time, 24 latents, six blocks) at 4 channels."""
    torch.manual_seed(seed)
    model = MiniMaxH3VideoEncoderModel(block_out_channels=(4,) * 6, norm_num_groups=2)
    for p in model.parameters():
        p.data = torch.randn_like(p) * 0.2
    return model.eval()


def test_the_statistics_cover_every_latent_channel_and_are_strictly_positive_in_scale() -> None:
    assert len(DEFAULT_LATENTS_MEAN) == len(DEFAULT_LATENTS_STD) == LATENTS
    assert min(DEFAULT_LATENTS_STD) > 0


def test_the_statistics_are_buffers_that_stay_out_of_the_state_dict() -> None:
    model = small_encoder()

    assert model.latents_mean.shape == (LATENTS,) and model.latents_mean.dtype == torch.float32
    assert not any("latents" in key for key in model.state_dict())


@pytest.mark.parametrize("frames, expected", [(5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (100, 90)])
def test_a_frame_count_snaps_down_to_the_seventeen_k_plus_five_grid(frames: int, expected: int) -> None:
    assert align_frame_count(frames) == expected


@pytest.mark.parametrize("frames", [0, 1, 4])
def test_clips_shorter_than_five_frames_are_refused(frames: int) -> None:
    with pytest.raises(ValueError, match="at least 5 frames, got"):
        align_frame_count(frames)


@pytest.mark.parametrize("frames, latents", [(1, 1), (5, 2), (22, 7), (39, 12), (56, 17)])
def test_latent_frame_counts_follow_five_per_seventeen_frames_plus_two(frames: int, latents: int) -> None:
    assert video_latent_frames(frames) == latents


def test_moments_carry_twice_the_latent_channels_at_sixteenth_resolution_and_quarter_time() -> None:
    model = small_encoder()

    moments = model.encode_moments(torch.rand(1, 3, 17, 32, 48))

    assert moments.shape == (1, 2 * LATENTS, 5, 2, 3)


def test_pixels_are_standardised_with_the_imagenet_statistics_before_the_first_conv() -> None:
    model = small_encoder()
    seen = []
    model.conv_in.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    pixels = torch.rand(1, 3, 1, 32, 32)

    model.encode_moments(pixels)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(seen[0], (pixels - mean) / std)


def test_half_precision_pixels_are_standardised_in_fp32_then_cast_to_the_weight_dtype() -> None:
    model = small_encoder()
    seen = []
    model.conv_in.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    pixels = torch.rand(1, 3, 1, 32, 32, dtype=torch.float16)

    model.encode_moments(pixels)

    assert seen[0].dtype == torch.float32  # conv_in weights are fp32 here
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(seen[0], (pixels.float() - mean) / std)


def test_normalising_moments_standardises_only_the_mean_half_and_drops_the_log_variance() -> None:
    model = small_encoder()
    mean_half = torch.randn(2, LATENTS, 1, 2, 2)
    moments = torch.cat([mean_half, torch.full_like(mean_half, 99.0)], dim=1).to(torch.float16)

    out = model._normalize(moments)

    expected = (mean_half.half().float() - model.latents_mean.view(1, -1, 1, 1, 1)) / model.latents_std.view(
        1, -1, 1, 1, 1
    )
    assert out.dtype == torch.float32 and out.shape == (2, LATENTS, 1, 2, 2)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("ndim", [4, 5])
def test_an_image_encodes_to_one_normalised_latent_frame_whether_or_not_it_has_a_time_axis(ndim: int) -> None:
    model = small_encoder()
    image = torch.rand(2, 3, 32, 32)

    out = model.encode_image(image if ndim == 4 else image.unsqueeze(2))

    assert out.shape == (2, LATENTS, 1, 2, 2)
    torch.testing.assert_close(out, model._normalize(model.encode_moments(image.unsqueeze(2))))


def test_a_single_frame_clip_is_encoded_as_an_image() -> None:
    model = small_encoder()
    clip = torch.rand(1, 3, 1, 32, 32)

    torch.testing.assert_close(model.encode_video(clip), model.encode_image(clip))


@pytest.mark.parametrize("frames", [5, 22, 39])
def test_clips_on_the_grid_encode_to_the_reference_latent_frame_count(frames: int) -> None:
    model = small_encoder()

    out = model.encode_video(torch.rand(1, 3, frames, 32, 32))

    assert out.shape == (1, LATENTS, video_latent_frames(frames), 2, 2)


def test_each_seventeen_frame_chunk_is_encoded_independently_of_the_rest() -> None:
    model = small_encoder()
    clip = torch.rand(1, 3, 34, 32, 32)
    changed = clip.clone()
    changed[:, :, VIDEO_CLIP_LENGTH:] = torch.rand(1, 3, 17, 32, 32)

    out, out_changed = model.encode_video(clip), model.encode_video(changed)

    first_chunk = model._normalize(model.encode_moments(clip[:, :, :VIDEO_CLIP_LENGTH]))
    torch.testing.assert_close(out[:, :, :5], first_chunk)
    torch.testing.assert_close(out[:, :, :5], out_changed[:, :, :5])
    assert not torch.allclose(out[:, :, 5:], out_changed[:, :, 5:])


def test_a_partial_last_chunk_is_padded_by_repeating_the_final_frame_and_the_last_three_latents_dropped() -> None:
    model = small_encoder()
    clip = torch.rand(1, 3, 22, 32, 32)
    padded = torch.cat([clip, clip[:, :, -1:].expand(-1, -1, 12, -1, -1)], dim=2)

    out = model.encode_video(clip)

    assert padded.shape[2] == 34
    reference = model._normalize(
        torch.cat([model.encode_moments(padded[:, :, :17]), model.encode_moments(padded[:, :, 17:])], dim=2)[:, :, :-3]
    )
    assert out.shape[2] == 7
    torch.testing.assert_close(out, reference)
