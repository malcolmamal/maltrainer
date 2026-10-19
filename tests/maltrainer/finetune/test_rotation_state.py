from __future__ import annotations

from typing import List

import torch
from torch import nn

from tests.maltrainer.finetune.rotation_model import rotator, step


def test_with_no_window_active_the_state_dict_holds_only_frozen_states() -> None:
    assert rotator().state_dict() == {"windows": ["attn", "mlp"], "optimizers": {}}


def test_a_window_resumes_its_optimizer_state_after_a_rotation() -> None:
    windows = rotator()
    windows.activate(0)
    step(windows)
    momentum = windows.optimizer.state_dict()["state"][0]["momentum_buffer"].clone()

    windows.activate(1)
    windows.activate(0)

    torch.testing.assert_close(windows.optimizer.state_dict()["state"][0]["momentum_buffer"], momentum)
    assert list(windows.optimizer_states) == [1]


def test_the_state_dict_holds_every_windows_optimizer_state_as_cpu_copies() -> None:
    windows = rotator()
    windows.activate(0)
    step(windows)
    windows.activate(1)

    state = windows.state_dict()

    assert state["windows"] == ["attn", "mlp"]
    assert sorted(state["optimizers"]) == [0, 1]
    assert state["optimizers"][0]["param_groups"][0]["params"] == [0, 1, 2, 3]
    assert state["optimizers"][1]["state"] == {}
    assert state["optimizers"][0]["state"][0]["momentum_buffer"].device.type == "cpu"


def test_the_active_windows_state_is_a_snapshot() -> None:
    windows = rotator()
    windows.activate(0)
    step(windows)

    state = windows.state_dict()
    snapshot = state["optimizers"][0]["state"][0]["momentum_buffer"].clone()
    step(windows)

    assert list(state["optimizers"]) == [0]
    assert torch.equal(state["optimizers"][0]["state"][0]["momentum_buffer"], snapshot)
    assert not torch.equal(windows.optimizer.state_dict()["state"][0]["momentum_buffer"], snapshot)


def test_a_loaded_state_is_used_when_its_window_activates() -> None:
    source = rotator()
    source.activate(0)
    step(source)
    saved = source.state_dict()
    windows = rotator()

    windows.load_state_dict(
        {"windows": saved["windows"], "optimizers": {str(k): v for k, v in saved["optimizers"].items()}}
    )
    windows.activate(0)

    assert sorted(windows.optimizer_states) == []
    torch.testing.assert_close(
        windows.optimizer.state_dict()["state"][2]["momentum_buffer"],
        saved["optimizers"][0]["state"][2]["momentum_buffer"],
    )


class _Handle:
    def __init__(self, removed: List[str], name: str) -> None:
        self.removed, self.name = removed, name

    def remove(self) -> None:
        self.removed.append(self.name)


def test_a_fused_optimizers_grad_hooks_are_removed_when_its_window_freezes() -> None:
    removed: List[str] = []

    def make_optimizer(params: List[nn.Parameter], name: str) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(params, lr=0.1)
        optimizer._hook_handles = [_Handle(removed, f"{name}0"), _Handle(removed, f"{name}1")]
        return optimizer

    windows = rotator(make_optimizer=make_optimizer)
    windows.activate(0)
    optimizer = windows.optimizer

    windows.deactivate()

    assert removed == ["attn0", "attn1"]
    assert optimizer._hook_handles == []
