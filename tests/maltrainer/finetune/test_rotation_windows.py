from __future__ import annotations

import logging

import pytest
import torch

from maltrainer.finetune.rotation import WindowRotator
from tests.maltrainer.finetune.rotation_model import WINDOWS, Model, Optimizers, master, rotator

ATTN_KEYS = ["blocks.0.attn.wq.weight", "blocks.0.attn.wk.weight", "blocks.1.attn.wq.weight", "blocks.1.attn.wk.weight"]
MLP_KEYS = ["blocks.0.mlp.weight", "blocks.1.mlp.weight"]


def test_windows_take_their_block_linears_in_model_order() -> None:
    windows = rotator()

    assert windows.names == ["attn", "mlp"]
    assert windows.windows == [("attn", ATTN_KEYS), ("mlp", MLP_KEYS)]
    assert windows.keys() == ATTN_KEYS + MLP_KEYS


def test_a_windows_size_is_its_weights_entries() -> None:
    windows = rotator()

    assert [windows.window_parameters(i) for i in range(2)] == [16, 12]


def test_a_window_may_list_several_prefixes() -> None:
    windows = rotator([("all", ("attn.wq", "mlp"))])

    assert windows.keys() == [
        "blocks.0.attn.wq.weight",
        "blocks.0.mlp.weight",
        "blocks.1.attn.wq.weight",
        "blocks.1.mlp.weight",
    ]


def test_nothing_is_active_or_built_before_the_first_activate() -> None:
    optimizers = Optimizers()

    windows = rotator(make_optimizer=optimizers)

    assert (windows.current, windows.optimizer, windows.optimizer_states, optimizers.made) == (None, None, {}, [])


def test_a_window_matching_no_linear_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"finetune window 'ffn' \(\['ffn.'\]\) matches no Linear"):
        rotator([("attn", ("attn.",)), ("ffn", ("ffn.",))])


def test_a_linear_in_two_windows_is_an_error() -> None:
    with pytest.raises(ValueError, match="blocks.0.attn.wq.weight is in two finetune windows, 'attn' and 'wq'"):
        rotator([("attn", ("attn.",)), ("wq", ("attn.wq",))])


def test_block_linears_left_out_of_every_window_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.finetune.rotation"):
        rotator([("attn", ("attn.",))])

    assert caplog.messages == [
        "2 block Linears are in no finetune window and stay frozen, e.g. ['blocks.0.mlp.weight', 'blocks.1.mlp.weight']"
    ]


def test_a_master_weight_outside_the_numbered_blocks_is_in_no_window(caplog: pytest.LogCaptureFixture) -> None:
    weights = {**master(), "head.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}

    with caplog.at_level(logging.WARNING, logger="maltrainer.finetune.rotation"):
        windows = WindowRotator(Model(), weights, WINDOWS, "cpu", torch.float32, Optimizers())

    assert "head.weight" not in windows.keys()
    assert caplog.messages == ["1 block Linears are in no finetune window and stay frozen, e.g. ['head.weight']"]


def test_full_coverage_of_the_blocks_warns_about_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.finetune.rotation"):
        rotator()

    assert caplog.messages == []


def test_a_saved_state_from_other_windows_is_refused() -> None:
    windows = rotator()

    with pytest.raises(ValueError, match=r"rotated windows \['attn'\], this config has \['attn', 'mlp'\]"):
        windows.load_state_dict({"windows": ["attn"], "optimizers": {}})


def test_states_cannot_be_loaded_once_a_window_is_active() -> None:
    windows = rotator()
    windows.activate(0)

    with pytest.raises(RuntimeError, match="before activating a window"):
        windows.load_state_dict({"windows": ["attn", "mlp"], "optimizers": {}})
