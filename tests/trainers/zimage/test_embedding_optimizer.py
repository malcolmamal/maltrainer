"""The textual-inversion optimizer: it owns the whole embedding table; Prodigy and its fallback."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler
from prodigyopt import Prodigy
from torch.optim import AdamW

from .runs import prepared_embedding
from .world import Workspace


def _trainer(ws: Workspace, **train: Any) -> Any:
    ws.add_images(1)
    return prepared_embedding(ws.embedding_config(train=train))


def test_the_defaults_are_adamw_at_5e_minus_4_over_the_input_embedding_table_alone(workspace: Workspace) -> None:
    trainer = _trainer(workspace, learning_rate=None)

    group = trainer.optimizer.param_groups[0]
    config = cast(Any, trainer.noise_scheduler).config
    assert type(trainer.optimizer) is AdamW
    assert group["params"] == [trainer.text_encoder.get_input_embeddings().weight]
    assert (group["lr"], group["betas"], group["weight_decay"], group["eps"]) == (5e-4, (0.9, 0.999), 0.01, 1e-8)
    assert isinstance(trainer.noise_scheduler, FlowMatchEulerDiscreteScheduler)
    assert (config.num_train_timesteps, config.shift) == (1000, 3.0)


def test_numbers_written_as_text_are_converted(workspace: Workspace) -> None:
    text = {"learning_rate": "2e-3", "adam_beta1": "0.8", "adam_beta2": 0.9, "weight_decay": "0.1", "adam_epsilon": 1}

    group = _trainer(workspace, **text).optimizer.param_groups[0]

    assert (group["lr"], group["betas"], group["weight_decay"], group["eps"]) == (2e-3, (0.8, 0.9), 0.1, 1.0)


def test_prodigy_gets_every_option_from_the_config(workspace: Workspace) -> None:
    options = {
        "optimizer": "prodigy",
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

    optimizer = _trainer(workspace, **options).optimizer

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
        ("prodigy_decouple", "decouple", None, True),
        ("prodigy_use_bias_correction", "use_bias_correction", "Y", True),
        ("prodigy_use_bias_correction", "use_bias_correction", "no", False),
        ("prodigy_slice_p", "slice_p", "3", 3),
        ("prodigy_slice_p", "slice_p", None, 1),
        ("prodigy_d0", "d0", None, 1e-6),
    ],
)
def test_prodigy_flags_and_numbers_are_read_leniently_and_default_when_unreadable(
    workspace: Workspace, option: str, key: str, value: Any, expected: Any
) -> None:
    group = _trainer(workspace, optimizer="prodigy", **{option: value}).optimizer.param_groups[0]

    assert group[key] == expected


def test_prodigy_that_cannot_be_imported_or_built_falls_back_to_adamw_with_a_warning(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no kernel")

    with caplog.at_level(logging.WARNING, logger="train_zimage_embedding"):
        monkeypatch.setitem(sys.modules, "prodigyopt", None)
        missing = _trainer(workspace, optimizer="prodigy").optimizer
        monkeypatch.setitem(sys.modules, "prodigyopt", SimpleNamespace(Prodigy=broken))
        failing = _trainer(workspace, optimizer="prodigy8", learning_rate=0.5).optimizer

    assert type(missing) is AdamW and type(failing) is AdamW and failing.param_groups[0]["lr"] == 0.5
    assert "Failed to load prodigy optimizer (prodigy)" in caplog.text and "no kernel" in caplog.text
