"""H3's Prodigy settings: the multiplier rule, safeguarded warmup and what an old release cannot take."""

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


def test_prodigy_takes_the_lora_defaults_and_drops_none_values(workspace: Workspace) -> None:
    trainer = _built(workspace, optimizer="Prodigy", adam_beta1=0.8, prodigy_d0=1e-5)

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert resolved["name"] == "prodigy" and resolved["lr"] == 1.0 and resolved["d_coef"] == 2.0
    assert resolved["betas"] == (0.8, 0.99) and resolved["d0"] == 1e-5 and "beta3" not in resolved
    assert resolved["decouple"] is True and resolved["safeguard_warmup"] is False


@pytest.mark.parametrize(
    ("train", "lr"),
    [
        ({"learning_rate": 1e-3}, 1.0),  # AdamW-scale: not meant as a multiplier
        ({"learning_rate": 0.5}, 0.5),
        ({"learning_rate": 1e-3, "prodigy_lr": 0.3}, 0.3),
    ],
)
def test_prodigys_multiplier_ignores_an_adamw_scale_learning_rate(
    workspace: Workspace, train: dict[str, Any], lr: float
) -> None:
    trainer = _built(workspace, optimizer="prodigy", **train)

    assert trainer.run_metadata["optimizer_resolved"]["lr"] == lr


def test_prodigy_safeguards_its_warmup_when_a_learning_rate_warmup_is_configured(workspace: Workspace) -> None:
    trainer = _built(workspace, optimizer="prodigy", learning_rate_warmup_steps=5, prodigy_slice_p=2, prodigy_beta3=0.9)

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert resolved["safeguard_warmup"] is True and resolved["slice_p"] == 2 and resolved["beta3"] == 0.9


def test_prodigy_in_a_low_precision_lora_warns_and_an_adamw_scale_rate_is_explained(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _built(workspace, lora={"dtype": "bfloat16"}, optimizer="prodigy", learning_rate=1e-4)

    assert "Set lora.dtype: float32" in caplog.text
    assert "is AdamW-scale, so Prodigy's multiplier is 1.0 instead" in caplog.text


def test_settings_an_old_prodigy_cannot_take_are_dropped_with_a_warning(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "prodigyopt", SimpleNamespace(Prodigy=_OldProdigy))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = _built(workspace, optimizer="prodigy", prodigy_slice_p=2)

    assert isinstance(trainer.optimizer, _OldProdigy)
    assert "does not accept ['growth_rate', 'slice_p']" in caplog.text
    assert "growth_rate" not in trainer.run_metadata["optimizer_resolved"]
