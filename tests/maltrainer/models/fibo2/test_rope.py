from __future__ import annotations

import math

import torch

from maltrainer.models.fibo2.rope import apply_rope, gist_ids, grid_ids, rope_angles

CPU = torch.device("cpu")


def test_grid_ids_hold_the_plane_the_row_and_the_column() -> None:
    ids = grid_ids(513, 2, 3, CPU)

    assert ids.tolist() == [[513, 0, 0], [513, 0, 1], [513, 0, 2], [513, 1, 0], [513, 1, 1], [513, 1, 2]]


def test_gist_ids_count_planes_from_one_with_row_and_column_zero() -> None:
    assert gist_ids(3, CPU).tolist() == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]


def test_position_zero_is_the_identity_rotation() -> None:
    cos, sin = rope_angles(torch.zeros(1, 3, dtype=torch.long))

    assert torch.all(cos == 1) and torch.all(sin == 0)
    x = torch.randn(1, 1, 2, 128)
    assert torch.equal(apply_rope(x, cos, sin), x)


def test_the_first_frequency_of_each_axis_rotates_by_the_position_in_radians() -> None:
    cos, sin = rope_angles(torch.tensor([[2, 3, 1]]))

    # frequency 0 of an axis is 256 ** 0 = 1; the axes hold 16, 24 and 24 frequencies
    assert [round(math.atan2(sin[0, 0, 0, i].item(), cos[0, 0, 0, i].item()), 5) for i in (0, 16, 40)] == [
        2.0,
        3.0,
        1.0,
    ]
    assert cos.shape == (1, 1, 1, 64)


def test_the_axes_use_their_own_dimension_for_the_frequencies() -> None:
    _, sin = rope_angles(torch.tensor([[0, 1, 0]]))

    # the row axis has 48 dims: frequency j is 256 ** (-2j / 48)
    expected = math.sin(256 ** (-2 * 1 / 48))
    assert sin[0, 0, 0, 16 + 1].item() == torch.tensor(expected).item()


def test_rotation_keeps_the_length_of_every_pair() -> None:
    cos, sin = rope_angles(grid_ids(5, 2, 2, CPU))
    x = torch.randn(1, 4, 2, 128)

    rotated = apply_rope(x, cos, sin)

    assert torch.allclose(rotated.unflatten(-1, (-1, 2)).norm(dim=-1), x.unflatten(-1, (-1, 2)).norm(dim=-1), atol=1e-5)


def test_a_pair_is_rotated_counter_clockwise() -> None:
    cos, sin = torch.zeros(1, 1, 1, 64), torch.ones(1, 1, 1, 64)
    x = torch.zeros(1, 1, 1, 128)
    x[..., 0] = 1.0  # (real, imaginary) = (1, 0) turned a quarter turn is (0, 1)

    rotated = apply_rope(x, cos, sin)

    assert rotated[..., :2].flatten().tolist() == [0.0, 1.0]


def test_the_rotation_is_rounded_once_to_the_input_dtype() -> None:
    cos, sin = rope_angles(grid_ids(5, 2, 2, CPU))
    x = torch.randn(1, 4, 2, 128).bfloat16()

    assert apply_rope(x, cos, sin).dtype == torch.bfloat16
