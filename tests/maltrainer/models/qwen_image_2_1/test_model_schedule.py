from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.qwen_image_2_1.model import apply_flow_shift, calculate_shift


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(256, 0.5), (8192, 0.9), (4096, 0.5 + 3840 / 19840), (0, 0.5 - 256 / 19840)],
)
def test_the_shift_grows_linearly_from_the_base_to_the_max_sequence_length(seq_len: int, expected: float) -> None:
    assert calculate_shift(seq_len) == pytest.approx(expected)


def test_the_shift_follows_custom_anchor_points() -> None:
    shift = calculate_shift(150, base_seq_len=100, max_seq_len=200, base_shift=1.0, max_shift=3.0)

    assert shift == pytest.approx(2.0)


def test_the_shift_is_a_plain_float() -> None:
    assert type(calculate_shift(1024)) is float


def test_a_zero_shift_returns_the_timesteps_untouched() -> None:
    t = torch.tensor([0.0, 0.3, 1.0])

    assert apply_flow_shift(t, mu=0.0) is t


def test_the_flow_shift_maps_the_unit_interval_onto_itself_and_bends_it_upward() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])

    shifted = apply_flow_shift(t, mu=math.log(2.0))

    torch.testing.assert_close(shifted, torch.tensor([0.0, 2.0 / 3.0, 1.0]))


def test_a_negative_shift_bends_the_timesteps_downward() -> None:
    shifted = apply_flow_shift(torch.tensor([0.5]), mu=-math.log(2.0))

    torch.testing.assert_close(shifted, torch.tensor([1.0 / 3.0]))


def test_the_flow_shift_keeps_the_dtype_and_stays_monotonic() -> None:
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)

    shifted = apply_flow_shift(t, mu=0.9)

    assert shifted.dtype == torch.float64
    assert bool((shifted[1:] > shifted[:-1]).all())
