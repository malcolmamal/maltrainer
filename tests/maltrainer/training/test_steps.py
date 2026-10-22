"""plan_training_steps and epoch_order, pinned value by value (they decide how long every run is)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from maltrainer.training.steps import epoch_order, plan_training_steps


@pytest.mark.parametrize(
    ("samples", "config", "expected"),
    [
        (10, {}, (10, 10, 100)),
        (10, {"num_epochs": 3}, (10, 3, 30)),
        (10, {"num_epochs": 3, "gradient_accumulation_steps": 4}, (3, 3, 7)),
        (10, {"max_steps": 50, "gradient_accumulation_steps": 2}, (5, 10, 50)),
        (10, {"max_steps": 1}, (10, 1, 1)),
        (10, {"max_steps": 25}, (10, 3, 25)),
        (0, {"max_steps": 7, "gradient_accumulation_steps": 2}, (1, 14, 7)),
        (0, {"num_epochs": 4}, (1, 4, 4)),
        (1, {"num_epochs": 1, "gradient_accumulation_steps": 8}, (1, 1, 1)),
        (10, {"max_steps": "6"}, (10, 1, 6)),
    ],
)
def test_the_plan_for_a_dataset_size_and_config(
    samples: int, config: dict[str, Any], expected: tuple[int, int, int]
) -> None:
    assert plan_training_steps(samples, config) == expected


@pytest.mark.parametrize("value", [0, None, -3])
def test_a_nonsense_accumulation_means_one(value: Any) -> None:
    assert plan_training_steps(6, {"num_epochs": 2, "gradient_accumulation_steps": value}) == (6, 2, 12)


@pytest.mark.parametrize("value", [0, None, -2])
def test_a_nonsense_epoch_count_means_one_and_max_steps_that_is_not_positive_is_ignored(value: Any) -> None:
    assert plan_training_steps(5, {"num_epochs": value, "max_steps": value}) == (5, 1, 5)


def test_epoch_order_visits_every_sample_repeats_times_in_the_shuffle_of_the_seed() -> None:
    random.seed(11)
    order = epoch_order(3, 2, 1, 0)
    random.seed(11)
    expected = [0, 0, 1, 1, 2, 2]
    random.shuffle(expected)

    assert order == expected
    assert sorted(order) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(("repeats", "count"), [(None, 4), (0, 4), (-2, 4), (3, 12), ("2", 8)])
def test_a_nonsense_repeat_count_means_one(repeats: Any, count: int) -> None:
    assert len(epoch_order(4, repeats, 1, 0)) == count


def test_each_process_takes_every_nth_sample_of_the_same_shuffle() -> None:
    random.seed(4)
    whole = epoch_order(6, 1, 1, 0)
    shares = []
    for index in range(3):
        random.seed(4)
        shares.append(epoch_order(6, 1, 3, index))

    assert shares == [whole[0::3], whole[1::3], whole[2::3]]
    assert sorted(i for share in shares for i in share) == list(range(6))
