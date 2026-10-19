from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.finetune.regularisation import epoch_order, set_step_scale


def test_every_subject_and_regularisation_image_appears_its_repeats() -> None:
    random.seed(0)

    order = epoch_order([0, 1], [5, 6, 7], repeats=3, reg_repeats=2)

    assert Counter(order) == {0: 3, 1: 3, 5: 2, 6: 2, 7: 2}


def test_subjects_appear_at_least_once_but_regularisation_can_be_off() -> None:
    random.seed(0)

    order = epoch_order([0, 1], [5], repeats=0, reg_repeats=-1)

    assert sorted(order) == [0, 1]


def test_the_order_is_shuffled_by_pythons_random() -> None:
    random.seed(3)
    first = epoch_order(range(10), range(10, 20), repeats=1, reg_repeats=1)
    random.seed(3)

    second = epoch_order(range(10), range(10, 20), repeats=1, reg_repeats=1)

    assert first == second
    assert first != sorted(first)


def _sgd() -> torch.optim.SGD:
    return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)


def test_the_step_scale_is_set_on_an_optimizer_that_reads_it() -> None:
    optimizer = _sgd()
    optimizer.lr_scale = 1.0

    set_step_scale(optimizer, 0.2)

    assert optimizer.lr_scale == 0.2


def test_the_step_scale_reaches_the_optimizer_inside_a_wrapper() -> None:
    inner = SimpleNamespace(lr_scale=1.0)
    wrapper = SimpleNamespace(optimizer=inner)

    set_step_scale(wrapper, 1)

    assert inner.lr_scale == 1.0
    assert isinstance(inner.lr_scale, float)
    assert not hasattr(wrapper, "lr_scale")


def test_an_optimizer_without_a_step_scale_is_an_error() -> None:
    with pytest.raises(ValueError, match="SGD has no lr_scale; regularisation images need Automagic3"):
        set_step_scale(_sgd(), 0.2)
