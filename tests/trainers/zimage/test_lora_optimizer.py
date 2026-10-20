"""Optimizer choice and value coercion, the 8-bit and Prodigy fallbacks, and the warmup scheduler."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler
from prodigyopt import Prodigy
from torch.optim import AdamW

from .runs import prepared_lora, run_lora
from .world import Workspace


def _optimizer(ws: Workspace, **train: Any) -> Any:
    ws.add_images(1)
    return prepared_lora(ws.lora_config(train=train)).optimizer


class Recorder:
    """Stands for a bitsandbytes optimizer: remembers how it was built."""

    def __init__(self, params: Any, **options: Any) -> None:
        self.params, self.options = list(params), options


def test_the_defaults_are_adamw_at_1e_minus_4_with_no_scheduler_and_the_flow_match_noise_schedule(
    workspace: Workspace,
) -> None:
    workspace.add_images(1)
    trainer = prepared_lora(workspace.lora_config(train={"learning_rate": None}))

    group = trainer.optimizer.param_groups[0]
    config = cast(Any, trainer.noise_scheduler).config
    assert type(trainer.optimizer) is AdamW and trainer.lr_scheduler is None
    assert (group["lr"], group["betas"], group["weight_decay"], group["eps"]) == (1e-4, (0.9, 0.999), 0.01, 1e-8)
    assert isinstance(trainer.noise_scheduler, FlowMatchEulerDiscreteScheduler)
    assert (config.num_train_timesteps, config.shift) == (1000, 3.0)


def test_numbers_written_as_text_are_converted_and_exclude_lora_zeroes_the_weight_decay(workspace: Workspace) -> None:
    text = {"learning_rate": "2e-3", "adam_beta1": "0.8", "adam_beta2": 0.9, "weight_decay": "0.1", "adam_epsilon": 1}

    group = _optimizer(workspace, **text).param_groups[0]
    free = _optimizer(workspace, weight_decay=0.1, weight_decay_exclude_lora=True).param_groups[0]

    assert (group["lr"], group["betas"], group["weight_decay"], group["eps"]) == (2e-3, (0.8, 0.9), 0.1, 1.0)
    assert free["weight_decay"] == 0.0


def test_prodigy_gets_every_option_from_the_config(workspace: Workspace) -> None:
    options = {
        "optimizer": "Prodigy",
        "prodigy_beta3": "0.95",
        "prodigy_decouple": False,
        "prodigy_use_bias_correction": "yes",
        "prodigy_safeguard_warmup": 1,
        "prodigy_d0": "1e-5",
        "prodigy_d_coef": 2,
        "prodigy_growth_rate": 1.5,
        "prodigy_slice_p": 2.0,
        "weight_decay": 0.2,
    }

    optimizer = _optimizer(workspace, **options)

    group = optimizer.param_groups[0]
    assert isinstance(optimizer, Prodigy)
    assert (group["beta3"], group["decouple"], group["use_bias_correction"]) == (0.95, False, True)
    assert (group["safeguard_warmup"], group["d0"], group["d_coef"]) == (True, 1e-5, 2.0)
    assert (group["growth_rate"], group["slice_p"], group["weight_decay"]) == (1.5, 2, 0.2)


@pytest.mark.parametrize(
    ("option", "key", "value", "expected"),
    [
        ("prodigy_decouple", "decouple", "off", False),
        ("prodigy_decouple", "decouple", "maybe", True),
        ("prodigy_decouple", "decouple", 0, False),
        ("prodigy_use_bias_correction", "use_bias_correction", "Y", True),
        ("prodigy_use_bias_correction", "use_bias_correction", "no", False),
        ("prodigy_use_bias_correction", "use_bias_correction", "unclear", False),
        ("prodigy_decouple", "decouple", None, True),
        ("prodigy_slice_p", "slice_p", "3", 3),
        ("prodigy_slice_p", "slice_p", None, 1),
        ("prodigy_d0", "d0", None, 1e-6),
    ],
)
def test_prodigy_flags_and_numbers_are_read_leniently_and_default_when_unreadable(
    workspace: Workspace, option: str, key: str, value: Any, expected: Any
) -> None:
    group = _optimizer(workspace, optimizer="prodigy", **{option: value}).param_groups[0]

    assert group[key] == expected


def test_prodigy_that_cannot_be_imported_or_built_falls_back_to_adamw_with_a_warning(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no kernel")

    with caplog.at_level(logging.WARNING, logger="train_zimage_lora"):
        monkeypatch.setitem(sys.modules, "prodigyopt", None)
        missing = _optimizer(workspace, optimizer="prodigy", weight_decay_exclude_lora=True)
        monkeypatch.setitem(sys.modules, "prodigyopt", SimpleNamespace(Prodigy=broken))
        failing = _optimizer(workspace, optimizer="prodigy8", learning_rate=0.5)

    assert type(missing) is AdamW and type(failing) is AdamW
    assert (missing.param_groups[0]["weight_decay"], failing.param_groups[0]["lr"]) == (0.0, 0.5)
    assert "Failed to load prodigy optimizer (prodigy)" in caplog.text and "no kernel" in caplog.text


@pytest.mark.parametrize(("name", "kind"), [("adamw8bit", "AdamW8bit"), ("adam8bit", "Adam8bit")])
def test_the_8_bit_optimizers_come_from_bitsandbytes_with_the_adam_options(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, name: str, kind: str
) -> None:
    optim = SimpleNamespace(AdamW8bit=type("AdamW8bit", (Recorder,), {}), Adam8bit=type("Adam8bit", (Recorder,), {}))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(optim=optim))

    optimizer = _optimizer(workspace, optimizer=name.upper(), learning_rate=0.1, weight_decay=0.3)

    assert type(optimizer).__name__ == kind and optimizer.params
    assert optimizer.options == {"lr": 0.1, "betas": (0.9, 0.999), "weight_decay": 0.3, "eps": 1e-8}


def test_8_bit_adam_that_cannot_be_loaded_falls_back_to_adamw_with_a_warning(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    with caplog.at_level(logging.WARNING, logger="train_zimage_lora"):
        optimizer = _optimizer(workspace, optimizer="adamw8bit", weight_decay=0.3)

    assert type(optimizer) is AdamW and optimizer.param_groups[0]["weight_decay"] == 0.3
    assert "Failed to load bitsandbytes optimizer (adamw8bit)" in caplog.text


def test_warmup_ramps_the_learning_rate_linearly_over_the_first_steps(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_lora(workspace.lora_config(train={"learning_rate_warmup_steps": 4}))

    log = (workspace.out / "training.log").read_text(encoding="utf-8")
    assert [line.rsplit("LR: ", 1)[1] for line in log.splitlines() if "Step:" in line] == [
        "5.00e-04",
        "7.50e-04",
        "1.00e-03",
    ]
    assert [trainer.lr_scheduler.lr_lambdas[0](s) for s in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]
