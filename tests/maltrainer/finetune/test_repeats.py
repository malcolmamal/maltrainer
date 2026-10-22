from __future__ import annotations

import random
from collections import Counter

import pytest

from maltrainer.finetune.repeats import folder_repeats, repeat_order, samples_per_epoch, unknown_keys


def test_an_exact_path_entry_wins_over_a_folder_name_entry() -> None:
    table = {"people/ben": 3, "ben": 2}

    assert folder_repeats("people/ben", table, 1) == 3.0


def test_a_folder_name_entry_matches_the_folder_anywhere() -> None:
    assert folder_repeats("people/emmawatson", {"emmawatson": 0.5}, 1) == 0.5


def test_the_folder_path_is_normalised_before_matching() -> None:
    assert folder_repeats("./people/ben/", {"people/ben": 3}, 1) == 3.0


def test_a_folder_with_no_entry_gets_the_default() -> None:
    assert folder_repeats("people/ann", {"ben": 3}, 2) == 2.0
    assert folder_repeats("people/ann", None, 1.5) == 1.5


def test_entries_matching_no_folder_are_reported_sorted() -> None:
    table = {"zed": 1, "people/ben": 3, "emmawatson": 2, "people/typo": 1}

    assert unknown_keys(table, ["people/ben", "celebs/emmawatson"]) == ["people/typo", "zed"]
    assert unknown_keys(None, ["people/ben"]) == []


@pytest.mark.parametrize(
    ("count", "repeats", "expected"),
    [(10, 1, 10), (10, 3, 30), (10, 0.5, 5), (10, 2.5, 25), (3, 0.5, 2), (10, 0, 0)],
)
def test_samples_per_epoch(count: int, repeats: float, expected: int) -> None:
    assert samples_per_epoch(count, repeats) == expected


def test_whole_repeats_show_every_index_that_many_times() -> None:
    random.seed(0)

    order = repeat_order([([0, 1, 2], 2), ([7], 1)])

    assert Counter(order) == {0: 2, 1: 2, 2: 2, 7: 1}


def test_a_fractional_repeat_adds_distinct_extra_samples() -> None:
    random.seed(0)

    counts = Counter(repeat_order([(range(10), 1.5)]))

    assert sorted(counts.values()) == [1] * 5 + [2] * 5


def test_a_fraction_below_one_draws_a_subset() -> None:
    random.seed(0)

    order = repeat_order([(range(10), 0.3)])

    assert len(order) == 3
    assert len(set(order)) == 3


def test_a_negative_repeat_counts_as_zero() -> None:
    assert repeat_order([([1, 2], -1.0)]) == []


def test_the_length_matches_samples_per_epoch() -> None:
    random.seed(1)
    groups = [(range(7), 2.5), (range(4), 0.5), (range(3), 1)]

    assert len(repeat_order(groups)) == sum(samples_per_epoch(len(i), r) for i, r in groups)
