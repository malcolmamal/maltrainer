"""The optimizer choices, their fallbacks when a library is missing, and the warmup schedule."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .runs import prepared
from .world import Workspace

LOGGER = "train_minimax_h3_lora"


def _built(ws: Workspace, lora: dict[str, Any] | None = None, **train: Any) -> Any:
    ws.add_images(1)
    return prepared(ws.config(train=train, lora=lora or {}))


@pytest.fixture
def no_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import bitsandbytes`` and ``import prodigyopt`` fail, as on a machine without them."""
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    monkeypatch.setitem(sys.modules, "prodigyopt", None)


class _OldProdigy(torch.optim.Optimizer):
    """A prodigyopt release whose signature has neither ``growth_rate`` nor ``slice_p``."""

    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
    ) -> None:
        defaults = {"lr": lr, "d": d0, "weight_decay": weight_decay, "betas": betas, "beta3": beta3, "eps": eps}
        defaults.update(d_coef=d_coef, decouple=decouple, use_bias_correction=use_bias_correction)
        defaults["safeguard_warmup"] = safeguard_warmup
        super().__init__(params, defaults)


class _Adam8bit(torch.optim.AdamW):
    """Stands in for bitsandbytes' 8-bit Adam (importing the real library takes seconds)."""


class _AdamW8bit(torch.optim.AdamW):
    """Stands in for bitsandbytes' 8-bit AdamW."""


@pytest.fixture
def fake_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "bitsandbytes", SimpleNamespace(optim=SimpleNamespace(AdamW8bit=_AdamW8bit, Adam8bit=_Adam8bit))
    )


@pytest.mark.usefixtures("fake_bitsandbytes")
@pytest.mark.parametrize(("name", "cls"), [("adamw8bit", _AdamW8bit), ("adam8bit", _Adam8bit)])
def test_the_8bit_optimizers_use_the_bitsandbytes_classes(workspace: Workspace, name: str, cls: type) -> None:
    trainer = _built(workspace, optimizer=name, learning_rate=2e-3)

    assert type(trainer.optimizer).__name__ == cls.__name__
    assert trainer.run_metadata["optimizer_resolved"] == {"name": name, "lr": 2e-3}


@pytest.mark.usefixtures("no_libraries")
@pytest.mark.parametrize("name", ["adamw8bit", "prodigy"])
def test_a_missing_library_falls_back_to_plain_adamw_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = _built(workspace, optimizer=name, learning_rate=2e-3)

    assert type(trainer.optimizer) is torch.optim.AdamW and trainer.optimizer.param_groups[0]["lr"] == 2e-3
    assert "not available" in caplog.text and "Falling back to AdamW" in caplog.text


@pytest.mark.parametrize("name", ["adamw", "something-unknown"])
def test_anything_else_is_torch_adamw(workspace: Workspace, name: str) -> None:
    trainer = _built(workspace, optimizer=name, learning_rate=3e-3)

    assert type(trainer.optimizer) is torch.optim.AdamW and trainer.optimizer.param_groups[0]["lr"] == 3e-3
    assert len(trainer.optimizer.param_groups[0]["params"]) == 32


@pytest.mark.parametrize("name", ["automagic3", "automagic", "automagic_v3"])
def test_automagic3_records_the_resolved_settings(workspace: Workspace, name: str) -> None:
    trainer = _built(workspace, optimizer=name, learning_rate=1e-3, weight_decay=0.1, automagic_polarity_history=4)

    assert trainer.run_metadata["optimizer_resolved"] == {
        "name": "automagic3",
        "lr": 1e-3,
        "min_lr": 1e-8,
        "max_lr": 1e3,
        "beta2": 0.999,
        "eps": 1e-30,
        "clip_threshold": 1.0,
        "weight_decay": 0.1,
        "polarity_history": 4,
        "fused": False,
    }


def test_automagic3_accepts_both_spellings_of_its_settings_and_the_prefixed_one_wins(workspace: Workspace) -> None:
    trainer = _built(workspace, optimizer="automagic3", min_lr=1e-5, automagic_min_lr=1e-4, max_lr=2.0, beta2=0.9)

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert (resolved["min_lr"], resolved["max_lr"], resolved["beta2"]) == (1e-4, 2.0, 0.9)


def test_without_warmup_there_is_no_scheduler(workspace: Workspace) -> None:
    assert _built(workspace).lr_scheduler is None


def test_warmup_ramps_the_rate_from_one_nth_up_to_the_learning_rate(workspace: Workspace) -> None:
    trainer = _built(workspace, learning_rate=1e-3, learning_rate_warmup_steps=4)
    seen = [trainer.optimizer.param_groups[0]["lr"]]

    for _ in range(5):
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        seen.append(trainer.optimizer.param_groups[0]["lr"])

    assert seen == pytest.approx([1e-3 / 4, 2e-3 / 4, 3e-3 / 4, 1e-3, 1e-3, 1e-3])
