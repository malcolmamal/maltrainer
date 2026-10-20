"""The [text | audio | video] packed layout: position ids, index sets, tags and the latent reshapes feeding it."""

from __future__ import annotations

import pytest
import torch

import train_minimax_h3_lora as h3
from maltrainer.models.minimax_h3 import packing
from maltrainer.models.minimax_h3.model import MiniMaxH3TokenTag

from .world import Workspace

THIRD = 5.0 / 3.0  # one temporal unit of the video clock: FRAME_RESCALE * 1 token


def _ids(*rows: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def test_text_audio_and_two_video_frames_get_the_comfy_positions() -> None:
    # 4x4 latent: sqrt(area) 4, so the h and w axes are [0, 16]; audio t runs on from the text length.
    ids = packing.build_packed_position_ids(text_len=2, latent_t=2, latent_h=4, latent_w=4, audio_t=2)

    expected = _ids(
        *[(0, 0, 0), (1, 0, 0)],  # text: t = token index
        *[(2, 0, 0), (3, 0, 0), (2, 0, 16), (3, 0, 16)],  # audio: left channel w=first column, right w=last
        *[(2, 0, 0), (2, 0, 16), (2, 16, 0), (2, 16, 16)],  # video frame 0, origin = text length
        *[(2 + THIRD, 0, 0), (2 + THIRD, 0, 16), (2 + THIRD, 16, 0), (2 + THIRD, 16, 16)],  # frame 1
    )
    assert ids.dtype == torch.float64
    torch.testing.assert_close(ids, expected)


def test_a_wide_latent_is_centred_on_the_area_normalised_grid() -> None:
    # sqrt(4 * 8) = 5.657: h spans 0.707 of it, w 1.414, so w overshoots the [0, 32) range on both sides.
    ids = packing.build_packed_position_ids(text_len=0, latent_t=1, latent_h=4, latent_w=8)

    h = [4.686292, 16.0]
    w = [-6.627417, 4.686292, 16.0, 27.313708]
    expected = _ids(*[(0.0, hh, ww) for hh in h for ww in w])
    torch.testing.assert_close(ids, expected)


def test_video_frames_advance_on_the_one_four_four_four_four_cycle() -> None:
    ids = packing.build_packed_position_ids(text_len=3, latent_t=7, latent_h=2, latent_w=2)

    spans = [1, 4, 4, 4, 4, 1]
    offsets = [0.0] + [THIRD * sum(spans[: i + 1]) for i in range(6)]
    torch.testing.assert_close(ids[3:, 0], 3 + torch.tensor(offsets, dtype=torch.float64))
    assert torch.count_nonzero(ids[3:, 1:]) == 0  # a 2x2 latent has one patch at the grid origin


def test_without_audio_the_audio_block_is_empty() -> None:
    ids = packing.build_packed_position_ids(text_len=1, latent_t=1, latent_h=2, latent_w=2)

    assert ids.shape == (2, 3)
    torch.testing.assert_close(ids, _ids((0, 0, 0), (1, 0, 0)))


def test_the_layout_splits_the_sequence_into_text_audio_and_video_with_their_tags_and_clocks(
    workspace: Workspace,
) -> None:
    trainer = h3.MiniMaxH3LoRATrainer(workspace.config())

    layout = trainer._build_packed_layout(2, 2, 4, 4, 2)

    assert layout["text_indices"].tolist() == [0, 1]
    assert layout["audio_indices"].tolist() == [2, 3, 4, 5]
    assert layout["video_indices"].tolist() == list(range(6, 14))
    tags = [MiniMaxH3TokenTag.TEXT] * 2 + [MiniMaxH3TokenTag.AUDIO] * 4 + [MiniMaxH3TokenTag.VIDEO] * 8
    assert layout["token_tags"].tolist() == tags
    assert layout["timestep_indices"].tolist() == [0, 0, 1, 1, 1, 1] + [0] * 8
    torch.testing.assert_close(layout["position_ids"], packing.build_packed_position_ids(2, 2, 4, 4, 2))


def test_layouts_are_cached_per_shape_signature(workspace: Workspace) -> None:
    trainer = h3.MiniMaxH3LoRATrainer(workspace.config())

    first = trainer._build_packed_layout(3, 1, 2, 2, 2)

    assert trainer._build_packed_layout(3, 1, 2, 2, 2) is first
    assert trainer._build_packed_layout(4, 1, 2, 2, 2) is not first
    assert trainer._build_packed_layout(3, 1, 2, 2, 0) is not first
    assert len(trainer._layout_cache) == 3


def test_a_layout_without_audio_has_no_audio_tokens(workspace: Workspace) -> None:
    trainer = h3.MiniMaxH3LoRATrainer(workspace.config())

    layout = trainer._build_packed_layout(2, 1, 4, 2, 0)

    assert layout["audio_indices"].numel() == 0
    assert layout["video_indices"].tolist() == [2, 3]
    assert layout["timestep_indices"].tolist() == [0, 0, 0, 0]


def test_patchify_groups_each_two_by_two_patch_channel_major() -> None:
    latent = torch.arange(2 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 2, 1, 2, 4)

    tokens = packing.patchify_video(latent)

    # channel 0 holds 0..7 and channel 1 holds 8..15 over a 2x4 plane; a patch is a 2x2 window.
    assert tokens.shape == (1, 2, 8)
    assert tokens[0, 0].tolist() == [0, 1, 4, 5, 8, 9, 12, 13]
    assert tokens[0, 1].tolist() == [2, 3, 6, 7, 10, 11, 14, 15]


def test_patchify_orders_tokens_by_frame_then_row_then_column() -> None:
    latent = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 1, 2, 4, 4)

    tokens = packing.patchify_video(latent)

    assert tokens.shape == (1, 8, 4)
    assert tokens[0, :, 0].tolist() == [0, 2, 8, 10, 16, 18, 24, 26]


def test_audio_rows_are_channel_major_left_then_right() -> None:
    latent = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4)  # [B, 32-like C=3, 2 ch, T=4]

    rows = packing.pack_audio_rows(latent)

    assert rows.shape == (1, 8, 3)
    assert rows[0, 0].tolist() == [0, 8, 16] and rows[0, 3].tolist() == [3, 11, 19]  # left t0 and t3
    assert rows[0, 4].tolist() == [4, 12, 20]  # right t0


@pytest.mark.parametrize(("t", "shift", "expected"), [(0.5, 1.0, 0.5), (0.5, 3.0, 0.75), (0.25, 12.0, 0.8)])
def test_the_flow_shift_maps_t_through_shift_t_over_one_plus_shift_minus_one_t(
    t: float, shift: float, expected: float
) -> None:
    assert packing.apply_flow_shift(torch.tensor(t), shift).item() == pytest.approx(expected)


def test_video_sigma_one_half_pairs_with_audio_sigma_one_fifth() -> None:
    assert packing.remap_sigma(torch.tensor(0.5)).item() == pytest.approx(0.2)
    assert packing.remap_sigma(torch.tensor(0.5), from_shift=3.0, to_shift=3.0).item() == pytest.approx(0.5)
    assert packing.remap_sigma(torch.tensor(1.0)).item() == pytest.approx(1.0)
