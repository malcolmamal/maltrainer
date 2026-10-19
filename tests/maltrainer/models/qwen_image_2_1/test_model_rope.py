from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.model import apply_rotary_emb_qwen, build_rope_frequencies


def _angles(text_len: int, h: int, w: int) -> torch.Tensor:
    return build_rope_frequencies(text_len, h, w, axes_dim=(2, 2, 2)).angle()


def test_the_frequencies_cover_the_text_and_image_tokens_with_half_the_axes_total() -> None:
    rotary = build_rope_frequencies(text_len=5, latent_h=3, latent_w=2)

    assert rotary.shape == (5 + 6, (16 + 56 + 56) // 2)
    assert rotary.dtype == torch.complex64
    torch.testing.assert_close(rotary.abs(), torch.ones(11, 64))


def test_text_tokens_advance_all_three_axes_together_by_their_index() -> None:
    angles = _angles(text_len=3, h=2, w=2)

    torch.testing.assert_close(angles[:3], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))


def test_every_image_token_sits_on_the_frame_after_the_text() -> None:
    angles = _angles(text_len=3, h=2, w=2)

    torch.testing.assert_close(angles[3:, 0], torch.full((4,), 3.0))


@pytest.mark.parametrize(
    ("h", "w", "rows", "cols"),
    [
        (2, 2, [-1.0, -1.0, 0.0, 0.0], [-1.0, 0.0, -1.0, 0.0]),
        (3, 2, [-2.0, -2.0, -1.0, -1.0, 0.0, 0.0], [-1.0, 0.0] * 3),
        (2, 3, [-1.0] * 3 + [0.0] * 3, [-2.0, -1.0, 0.0] * 2),
        (1, 1, [-1.0], [-1.0]),
    ],
)
def test_image_rows_and_columns_are_centred_with_the_larger_half_below_zero(
    h: int, w: int, rows: list[float], cols: list[float]
) -> None:
    angles = _angles(text_len=1, h=h, w=w)

    torch.testing.assert_close(angles[1:, 1], torch.tensor(rows))
    torch.testing.assert_close(angles[1:, 2], torch.tensor(cols))


def test_each_axis_rotates_its_pairs_at_a_geometric_ladder_of_frequencies() -> None:
    rotary = build_rope_frequencies(text_len=2, latent_h=1, latent_w=1, axes_dim=(4, 2, 2))

    expected = torch.polar(torch.ones(2), torch.tensor([1.0, 0.01]) * 1.0)
    torch.testing.assert_close(rotary[1, :2], expected)
    torch.testing.assert_close(rotary[2, :2], torch.polar(torch.ones(2), torch.tensor([2.0, 0.02])))


def test_rotating_a_pair_by_an_angle_turns_it_counter_clockwise_and_keeps_its_length() -> None:
    x = torch.tensor([[[[1.0, 0.0, 0.0, 2.0]]]])
    freqs = torch.polar(torch.ones(1, 2), torch.tensor([[torch.pi / 2, torch.pi]]))

    rotated = apply_rotary_emb_qwen(x, freqs)

    torch.testing.assert_close(rotated, torch.tensor([[[[0.0, 1.0, 0.0, -2.0]]]]), atol=1e-6, rtol=0)


def test_unit_frequencies_leave_the_queries_unchanged_and_the_dtype_is_kept() -> None:
    x = torch.randn(2, 3, 4, 8).to(torch.bfloat16)

    rotated = apply_rotary_emb_qwen(x, torch.ones(3, 4, dtype=torch.complex64))

    assert rotated.dtype == torch.bfloat16
    torch.testing.assert_close(rotated, x)


def test_the_rotation_depends_on_the_token_position() -> None:
    x = torch.ones(1, 2, 1, 2)
    freqs = torch.polar(torch.ones(2, 1), torch.tensor([[0.0], [torch.pi / 2]]))

    rotated = apply_rotary_emb_qwen(x, freqs)

    torch.testing.assert_close(rotated[0, :, 0], torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), atol=1e-6, rtol=0)
