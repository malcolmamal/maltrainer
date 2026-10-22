from __future__ import annotations

import pytest
from torch import nn

from maltrainer.finetune.selection import select_trainable


def _model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2), nn.Linear(2, 2, bias=False))


def _trainable(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def test_without_prefixes_every_parameter_trains() -> None:
    model = _model()

    selected = select_trainable(model)

    names = ["0.weight", "0.bias", "1.weight", "1.bias", "2.weight"]
    assert [name for name, _ in selected] == names
    assert _trainable(model) == names
    assert selected[0][1] is model[0].weight


def test_only_parameters_under_a_prefix_train() -> None:
    model = _model()

    selected = select_trainable(model, prefixes=["1.", "2."])

    assert [name for name, _ in selected] == ["1.weight", "1.bias", "2.weight"]
    assert _trainable(model) == ["1.weight", "1.bias", "2.weight"]


def test_linears_only_leaves_biases_and_norm_scales_frozen() -> None:
    model = _model()

    selected = select_trainable(model, linears_only=True)

    assert [name for name, _ in selected] == ["0.weight", "2.weight"]
    assert _trainable(model) == ["0.weight", "2.weight"]


def test_selecting_again_freezes_what_the_last_selection_trained() -> None:
    model = _model()
    select_trainable(model)

    select_trainable(model, prefixes=["0."])

    assert _trainable(model) == ["0.weight", "0.bias"]


@pytest.mark.parametrize(
    ("prefixes", "linears_only", "message"),
    [
        (["missing."], False, r"\['missing.'\] matched no parameter"),
        (["1."], True, r"\['1.'\] with linears_only matched no parameter"),
    ],
)
def test_a_selection_matching_nothing_is_an_error(prefixes: list[str], linears_only: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_trainable(_model(), prefixes=prefixes, linears_only=linears_only)
