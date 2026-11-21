"""The INT8 window rotator's resume side: trained weights restored quantized as they were, optimizer states kept."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from .int8_model import Model, Optimizers, rotator, windows

QKV0, FC0, QKV1, FC1 = (
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.1.attn.qkv_proj.weight",
    "blocks.1.mlp.fc1.weight",
)


def _linear(model: nn.Module, key: str) -> nn.Linear:
    linear = model.get_submodule(key.removesuffix(".weight"))
    assert isinstance(linear, nn.Linear)
    return linear


def test_a_resumed_run_gets_its_trained_weights_back_quantized_as_they_were() -> None:
    first = Model()
    rot = rotator(first)
    rot.activate(0)
    with torch.no_grad():
        _linear(first, QKV0).weight.add_(0.5)
    rot.activate(1)
    resumed = Model()
    again = rotator(resumed)

    again.restore(rot.touched())

    for key in (QKV0, FC0):
        assert torch.equal(_linear(resumed, key).weight, _linear(first, key).weight)
        assert torch.equal(_linear(resumed, key).scale_weight, _linear(first, key).scale_weight)
        assert torch.equal(again.master[key], rot.weight(key))


def test_trained_weights_cannot_be_restored_into_a_running_window() -> None:
    rot = rotator()
    rot.activate(0)

    with pytest.raises(RuntimeError, match="before activating"):
        rot.restore({})


def test_each_windows_optimizer_state_is_kept_while_it_is_frozen_and_comes_back() -> None:
    model, optimizers = Model(), Optimizers()
    rot = rotator(model, make_optimizer=optimizers)
    rot.activate(0)
    for param in rot.optimizer.param_groups[0]["params"]:
        param.grad = torch.ones_like(param)
    rot.optimizer.step()
    momentum = [rot.optimizer.state[p]["momentum_buffer"].clone() for p in rot.optimizer.param_groups[0]["params"]]

    rot.activate(1)
    rot.activate(0)

    restored = [rot.optimizer.state[p]["momentum_buffer"] for p in rot.optimizer.param_groups[0]["params"]]
    assert all(torch.equal(a, b) for a, b in zip(momentum, restored, strict=True))
    assert set(rot.state_dict()["optimizers"]) == {0, 1}


def test_a_saved_state_loads_into_a_rotator_with_the_same_windows_only() -> None:
    rot = rotator()
    rot.activate(0)
    saved = rot.state_dict()

    fresh = rotator()
    fresh.load_state_dict(saved)
    assert set(fresh.optimizer_states) == {0}

    with pytest.raises(ValueError, match="rotated windows"):
        rotator(plan=[("only", windows(Model())[0][1])]).load_state_dict(saved)
    running = rotator()
    running.activate(0)
    with pytest.raises(RuntimeError, match="before activating"):
        running.load_state_dict(saved)


def test_a_weight_in_two_windows_is_refused() -> None:
    model = Model()
    both = [("a", [QKV0]), ("b", [QKV0, FC0])]

    with pytest.raises(ValueError, match="two finetune windows"):
        rotator(model, both)


def test_window_sizes_count_the_weights_whether_int8_or_dense() -> None:
    rot = rotator()

    assert rot.window_parameters(0) == 6 * 8 + 4 * 8
    rot.activate(0)
    assert rot.window_parameters(0) == 6 * 8 + 4 * 8 and rot.keys() == [QKV0, FC0, QKV1, FC1]
