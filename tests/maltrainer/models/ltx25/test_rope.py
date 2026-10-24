from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.ltx25.rope import (
    apply_rope,
    audio_positions,
    frequency_grid,
    make_rope,
    video_grid,
    video_positions,
)

CPU = torch.device("cpu")


def test_the_frequencies_run_geometrically_from_pi_over_two_to_theta_pi_over_two() -> None:
    freqs = frequency_grid(10000.0, 3, 96, True, CPU)

    assert freqs.shape == (16,) and freqs.dtype == torch.float32
    assert freqs[0].item() == pytest.approx(math.pi / 2) and freqs[-1].item() == pytest.approx(
        10000 * math.pi / 2, rel=1e-6
    )
    assert (freqs[1:] / freqs[:-1]).std().item() < 1e-4


def test_the_video_grid_numbers_every_latent_pixel_by_frame_row_and_column() -> None:
    grid = video_grid(2, 2, 3, CPU)

    assert grid.shape == (1, 3, 12, 2)
    assert grid[0, :, 7, 0].tolist() == [1.0, 0.0, 1.0] and grid[0, :, 7, 1].tolist() == [2.0, 1.0, 2.0]


def test_video_positions_are_in_pixels_with_a_causal_first_frame_and_seconds_in_time() -> None:
    pos = video_positions(3, 1, 1, (8, 32, 32), True, 24.0, CPU)

    # frame f starts at max(0, 8 f - 7) pixels-frames: 0, 1, 9; then divided by the frame rate
    assert pos[0, 0, :, 0].tolist() == pytest.approx([0.0, 1 / 24, 9 / 24])
    assert pos[0, 0, :, 1].tolist() == pytest.approx([1 / 24, 9 / 24, 17 / 24])
    assert pos[0, 1, 0].tolist() == [0.0, 32.0]


def test_video_positions_without_the_causal_fix_start_at_whole_strides() -> None:
    pos = video_positions(2, 1, 1, (8, 32, 32), False, 8.0, CPU)

    assert pos[0, 0, :, 0].tolist() == pytest.approx([0.0, 1.0])


def test_audio_positions_are_seconds_with_a_causal_first_latent() -> None:
    pos = audio_positions(3, CPU)

    # mel frame of latent f: max(0, 4 f - 3); one mel frame is 160 / 16000 s
    assert pos.shape == (1, 1, 3, 2)
    assert pos[0, 0, :, 0].tolist() == pytest.approx([0.0, 0.01, 0.05])
    assert pos[0, 0, :, 1].tolist() == pytest.approx([0.01, 0.05, 0.09])


def test_the_audio_positions_without_causality_start_at_zero_and_step_four_mel_frames() -> None:
    pos = audio_positions(2, CPU, causal=False)

    assert pos[0, 0, :, 0].tolist() == pytest.approx([0.0, 0.04])


def test_a_rope_has_a_rotation_per_token_and_head() -> None:
    grid = video_positions(2, 2, 2, (8, 32, 32), True, 24.0, CPU)

    cos, sin, split = make_rope(grid, 96, 2, (20, 64, 64), 10000.0, True, True, True)

    assert cos.shape == sin.shape == (1, 8, 2, 24) and split


def test_a_split_rope_pads_the_front_with_the_identity_rotation() -> None:
    grid = video_positions(1, 2, 2, (8, 32, 32), True, 24.0, CPU)

    cos, sin, _ = make_rope(grid, 104, 1, (20, 64, 64), 10000.0, True, True, True)

    # 3 axes x (104 // 6 = 17) = 51 frequencies in a half-width of 52: one padded in front
    assert cos.shape == (1, 4, 1, 52)
    assert torch.all(cos[..., 0] == 1) and torch.all(sin[..., 0] == 0)


def test_an_interleaved_rope_pads_by_the_remainder_of_the_width() -> None:
    grid = video_positions(1, 2, 2, (8, 32, 32), True, 24.0, CPU)

    cos, _, split = make_rope(grid, 104, 1, (20, 64, 64), 10000.0, False, True, False)

    assert cos.shape[-1] == 52 and not split  # 104 % 6 = 2 -> one padded rotation


def test_the_middle_of_a_patch_or_its_start_decides_the_position() -> None:
    grid = audio_positions(3, CPU)

    middle = make_rope(grid, 48, 2, (20,), 10000.0, True, True, True)
    start = make_rope(grid, 48, 2, (20,), 10000.0, True, True, False)

    assert not torch.allclose(middle[0], start[0])


def test_a_zero_position_rotates_by_the_lowest_frequency_pi_over_two_times_minus_one() -> None:
    grid = torch.zeros(1, 1, 1)

    cos, sin, _ = make_rope(grid, 8, 1, (20,), 10000.0, True, True, False)

    # the fractional position 0 maps to -1 in (2 p - 1): angle = -freq
    assert cos[0, 0, 0, 0].item() == pytest.approx(math.cos(math.pi / 2), abs=1e-6)
    assert sin[0, 0, 0, 0].item() == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("split", [True, False])
def test_a_quarter_turn_swaps_the_two_halves_or_neighbours(split: bool) -> None:
    cos, sin = torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    out = apply_rope(x, (cos, sin, split))

    # split pairs (x0, x2), (x1, x3); interleaved pairs (x0, x1), (x2, x3); a quarter turn maps (a, b) to (-b, a)
    expected = [-3.0, -4.0, 1.0, 2.0] if split else [-2.0, 1.0, -4.0, 3.0]
    assert out[0, 0].tolist() == expected


def test_the_identity_rotation_leaves_values_alone_and_the_dtype_is_kept() -> None:
    cos, sin = torch.ones(1, 3, 2, 4), torch.zeros(1, 3, 2, 4)
    x = torch.randn(2, 3, 16).bfloat16()

    out = apply_rope(x, (cos, sin, True))

    assert out.dtype == torch.bfloat16 and torch.equal(out, x)
