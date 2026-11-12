"""The H3 Image finetune's windows and always-trained modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3_image.finetune import always_on_params, plan_windows
from maltrainer.models.minimax_h3_image.finetune_int8 import linear_forward
from tests.maltrainer.finetune.int8_model import Model, int8_linear

LEAVES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _keys(blocks: int = 12) -> list[str]:
    return [f"blocks.{n}.{leaf}.weight" for n in range(blocks) for leaf in LEAVES]


def test_windows_are_runs_of_five_blocks_by_default_in_block_order() -> None:
    windows = plan_windows(_keys(), {})

    assert [name for name, _ in windows] == ["blocks 0-4", "blocks 5-9", "blocks 10-11"]
    assert [len(keys) for _, keys in windows] == [20, 20, 8]
    assert windows[0][1][:4] == [f"blocks.0.{leaf}.weight" for leaf in LEAVES]


def test_a_window_length_in_blocks_and_a_block_range_narrow_the_plan() -> None:
    windows = plan_windows(_keys(), {"blocks_per_window": 2, "block_range": [3, 8]})

    assert [name for name, _ in windows] == ["blocks 3-4", "blocks 5-6", "blocks 7-8"]


def test_targets_choose_which_linears_of_a_block_train() -> None:
    windows = plan_windows(_keys(2), {"targets": ["mlp.fc1"], "blocks_per_window": 1})

    assert [keys for _, keys in windows] == [["blocks.0.mlp.fc1.weight"], ["blocks.1.mlp.fc1.weight"]]


def test_keys_outside_the_numbered_blocks_never_train_in_a_window() -> None:
    windows = plan_windows(["token_refiner.blocks.0.attn.qkv_proj.weight", *_keys(1)], {})

    assert windows[0][1] == _keys(1)


@pytest.mark.parametrize("finetune", [{"blocks_per_window": -1}, {"targets": ["nothing"]}, {"block_range": [90, 99]}])
def test_a_plan_that_trains_nothing_or_makes_no_sense_is_refused(finetune: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"blocks_per_window|match no INT8 Linear"):
        plan_windows(_keys(2), finetune)


def test_always_on_weights_become_trainable_and_are_named_from_the_model() -> None:
    model = Model()

    trained = always_on_params(model, ["head"])

    assert [name for name, _ in trained] == ["head.weight"] and trained[0][1].requires_grad
    assert always_on_params(model, []) == []


def test_an_int8_block_linear_or_an_f16_weight_cannot_be_always_on() -> None:
    model = Model()
    model.add_module("half_head", nn.Linear(4, 2).half())

    with pytest.raises(ValueError, match="windows train it"):
        always_on_params(model, ["blocks"])
    with pytest.raises(ValueError, match="bf16 or fp32"):
        always_on_params(model, ["half_head"])


def test_a_module_without_a_linear_under_it_is_refused() -> None:
    model = nn.Sequential(nn.ReLU())

    with pytest.raises(ValueError, match="matched no Linear"):
        always_on_params(model, ["0"])


def test_a_linear_runs_on_its_int8_codes_or_on_a_dense_trained_weight() -> None:
    weight = torch.randn(6, 8, generator=torch.Generator().manual_seed(0))
    linear = int8_linear(weight)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))

    on_codes = linear_forward(linear, x)
    linear.weight = nn.Parameter(weight)
    on_dense = linear_forward(linear, x)

    torch.testing.assert_close(on_dense, x @ weight.T)
    torch.testing.assert_close(on_codes, on_dense, atol=0.2, rtol=0.1)
