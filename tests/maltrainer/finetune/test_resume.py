from __future__ import annotations

import pytest

from maltrainer.finetune.resume import check_same_trainable_set

NAMES = ["layers.0.proj.q.weight", "layers.0.proj.k.weight", "layers.1.proj.q.weight"]


def test_a_state_for_the_same_groups_and_parameters_is_accepted() -> None:
    check_same_trainable_set({"group_names": ["attn", "mlp"], "param_names": NAMES}, ["attn", "mlp"], NAMES)


def test_other_optimizer_groups_are_refused_first() -> None:
    state = {"group_names": ["attn", "mlp"], "param_names": NAMES}

    with pytest.raises(ValueError, match=r"had groups \['attn', 'mlp'\], this config builds \['all'\]"):
        check_same_trainable_set(state, ["all"], ["something.else"])


def test_other_parameters_in_the_same_groups_are_refused_with_what_differs() -> None:
    state = {"group_names": ["all"], "param_names": NAMES}
    now = [NAMES[0], "layers.5.proj.q.weight"]

    with pytest.raises(ValueError, match=r"trained 3 parameters, this config trains 2") as refusal:
        check_same_trainable_set(state, ["all"], now)

    assert "layers.1.proj.q.weight" in str(refusal.value)
    assert "layers.5.proj.q.weight" in str(refusal.value)


def test_the_same_parameters_in_another_order_are_refused() -> None:
    state = {"group_names": ["all"], "param_names": NAMES}

    with pytest.raises(ValueError, match="trained 3 parameters"):
        check_same_trainable_set(state, ["all"], list(reversed(NAMES)))


def test_a_state_without_recorded_names_is_taken_as_it_is() -> None:
    check_same_trainable_set({"group_names": ["all"]}, ["all"], NAMES)
