"""How an H3 epoch is ordered: shuffled with the run seed, repeated, and shared between processes."""

from __future__ import annotations

import random
from typing import Any

import pytest

from .runs import cached
from .world import Workspace


def _constant(value: int) -> property:
    return property(lambda _self: value)


def test_the_order_is_shuffled_with_the_run_seed(workspace: Workspace) -> None:
    workspace.add_images(6)
    first, second = cached(workspace.config()), cached(workspace.config())

    random.seed(5)
    one = first._epoch_order()
    random.seed(5)
    two = second._epoch_order()

    assert one == two and sorted(one) == list(range(6)) and one != list(range(6))


@pytest.mark.parametrize(("repeats", "expected"), [(3, 6), (1, 2), (0, 2), (None, 2)])
def test_repeats_multiply_the_epoch_and_a_nonsense_value_means_one(
    workspace: Workspace, repeats: Any, expected: int
) -> None:
    workspace.add_images(2)
    trainer = cached(workspace.config(dataset={"repeats": repeats}))

    order = trainer._epoch_order()

    assert len(order) == expected and sorted(set(order)) == [0, 1]


def test_each_process_trains_its_own_stride_of_the_shuffled_epoch(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(5)
    trainer = cached(workspace.config())
    monkeypatch.setattr(type(trainer.accelerator), "num_processes", _constant(2))
    shares = []

    for rank in (0, 1):
        monkeypatch.setattr(type(trainer.accelerator), "process_index", _constant(rank))
        random.seed(9)
        shares.append(trainer._epoch_order())

    assert [len(s) for s in shares] == [3, 2] and sorted(shares[0] + shares[1]) == [0, 1, 2, 3, 4]
