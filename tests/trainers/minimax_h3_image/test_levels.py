"""Which noise level a step runs at, and the likeness-optimization share per epoch."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3_image.grid import PDD_SIGMAS
from maltrainer.trainers.minimax_h3_image.levels import clean_share, draw_level

LEVELS = PDD_SIGMAS[:-1]
DRAWS = 6000


def _counts(share: float | None, levels: tuple[float, ...] = LEVELS) -> list[int]:
    torch.manual_seed(0)
    counts = [0] * len(levels)
    for _ in range(DRAWS):
        counts[draw_level(levels, share)] += 1
    return counts


def test_without_likeness_optimization_there_is_no_share() -> None:
    assert [clean_share(epoch, False) for epoch in (1, 9, 30)] == [None, None, None]


@pytest.mark.parametrize(
    ("epoch", "share"),
    [(1, 0.75), (8, 0.75), (9, 0.5), (10, 0.75), (11, 0.5), (12, 0.75)],
)
def test_likeness_optimization_holds_75_for_8_epochs_then_alternates_50_and_75(epoch: int, share: float) -> None:
    assert clean_share(epoch, True) == share


def test_the_even_draw_visits_every_level_about_equally() -> None:
    assert all(abs(count - DRAWS / 6) < 0.25 * DRAWS / 6 for count in _counts(None))


def test_a_share_puts_that_fraction_of_steps_on_levels_below_sigma_half() -> None:
    counts = _counts(0.75)

    assert abs(counts[-1] / DRAWS - 0.75) < 0.03  # sigma 0.232 is the only level under 0.5
    assert all(abs(count / DRAWS - 0.05) < 0.02 for count in counts[:-1])  # the other five split the rest


def test_levels_with_nothing_on_one_side_fall_back_to_the_even_draw() -> None:
    assert all(count > 0 for count in _counts(0.75, (1.0, 0.9, 0.8)))
    assert all(count > 0 for count in _counts(0.75, (0.4, 0.3, 0.2)))
