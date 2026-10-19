from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3.video_vae import MiniMaxH3VideoEncoderModel, video_latent_frames
from tests.maltrainer.models.minimax_h3.h3_video import MEAN, STD, clip, tiny_video_encoder


@pytest.mark.parametrize("frames", [5, 17, 22, 39])
def test_a_clip_is_cut_into_17_frame_chunks_of_five_latents_and_the_last_three_are_dropped(frames: int) -> None:
    chunks = -(-frames // 17)

    out = tiny_video_encoder().encode_video(clip(frames))

    assert out.shape == (1, 2, chunks * 5 - 3, 2, 2)


@pytest.mark.parametrize("frames", [5, 22, 39])
def test_aligned_clips_give_the_latent_count_the_dataset_expects(frames: int) -> None:
    assert tiny_video_encoder().encode_video(clip(frames)).shape[2] == video_latent_frames(frames)


def test_a_single_frame_is_encoded_as_an_image() -> None:
    model = tiny_video_encoder()
    frames = clip(1)

    video = model.encode_video(frames)

    assert video.shape == (1, 2, 1, 2, 2)
    torch.testing.assert_close(video, model.encode_image(frames[:, :, 0]))


def test_an_image_without_a_frame_axis_gets_one() -> None:
    model = tiny_video_encoder()
    image = clip(1)[:, :, 0]

    torch.testing.assert_close(model.encode_image(image), model.encode_image(image.unsqueeze(2)))


def test_the_chunks_are_encoded_independently_of_each_other() -> None:
    model = tiny_video_encoder()
    long = clip(34)

    both = model.encode_video(long)
    first = model.encode_video(long[:, :, :17])
    second_alone = model.encode_moments(long[:, :, 17:])

    torch.testing.assert_close(both[:, :, :2], first)
    torch.testing.assert_close(both[:, :, 5:], model._normalize(second_alone)[:, :, :2])


def test_a_short_tail_is_padded_by_repeating_the_last_frame() -> None:
    model = tiny_video_encoder()
    short = clip(5)
    padded = torch.cat([short, short[:, :, -1:].expand(-1, -1, 12, -1, -1)], dim=2)

    torch.testing.assert_close(model.encode_video(short), model.encode_video(padded))


def test_latents_are_the_posterior_mean_minus_the_channel_mean_over_the_channel_std() -> None:
    model = tiny_video_encoder()
    moments = torch.randn(1, 4, 1, 1, 1, generator=torch.Generator().manual_seed(3))

    out = model._normalize(moments)

    expected = (moments[:, :2] - torch.tensor(MEAN).view(1, 2, 1, 1, 1)) / torch.tensor(STD).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(out, expected)


def test_pixels_are_normalised_with_the_imagenet_statistics_before_the_first_conv() -> None:
    model = tiny_video_encoder()
    seen: list[torch.Tensor] = []
    model.conv_in.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    pixels = torch.rand(1, 3, 5, 16, 16, generator=torch.Generator().manual_seed(1))

    model.encode_moments(pixels)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(seen[0], (pixels - mean) / std)


def test_the_statistics_are_not_saved_with_the_weights() -> None:
    assert not {"latents_mean", "latents_std"} & set(tiny_video_encoder().state_dict())


def test_the_released_configuration_maps_17_frames_and_16_pixels_to_5_frames_and_one_latent_pixel() -> None:
    with torch.device("meta"):
        model = MiniMaxH3VideoEncoderModel()
        moments = model.encode_moments(torch.empty(1, 3, 17, 32, 48))
        latents = model.encode_video(torch.empty(1, 3, 39, 32, 48))

    assert moments.shape == (1, 48, 5, 2, 3)
    assert latents.shape == (1, 24, 12, 2, 3)
    assert len(model.down_blocks) == 6
