from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from maltrainer.optimizers.prodigy import build_prodigy


class FakeProdigy(torch.optim.Optimizer):
    """Records the settings it was built with; its signature is what the installed release accepts."""

    built: dict[str, Any] = {}  # noqa: RUF012 - a test double's record

    def __init__(
        self, params: Any, lr: float = 1.0, d0: float = 1e-6, betas: tuple[float, float] = (0.9, 0.999)
    ) -> None:
        FakeProdigy.built = {"lr": lr, "d0": d0, "betas": betas}
        super().__init__(params, {"lr": lr})


@pytest.fixture
def fake_prodigy(monkeypatch: pytest.MonkeyPatch) -> type[FakeProdigy]:
    module = ModuleType("prodigyopt")
    module.Prodigy = FakeProdigy  # type: ignore[attr-defined]  # a stand-in module
    monkeypatch.setitem(sys.modules, "prodigyopt", module)
    return FakeProdigy


def _params() -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.zeros(2))]


def test_only_the_settings_the_installed_release_accepts_are_passed(fake_prodigy: type[FakeProdigy]) -> None:
    optimizer, used = build_prodigy(_params(), {"prodigy_d0": 1e-5, "adam_beta1": 0.8}, lr=1e-4, weight_decay=0.1)

    assert isinstance(optimizer, FakeProdigy)
    assert used == {"lr": 1.0, "betas": (0.8, 0.999), "d0": 1e-5}
    assert fake_prodigy.built == used


@pytest.mark.parametrize(
    ("lr", "config", "expected"), [(1e-4, {}, 1.0), (0.5, {}, 0.5), (1e-4, {"prodigy_lr": 0.3}, 0.3)]
)
@pytest.mark.usefixtures("fake_prodigy")
def test_the_learning_rate_is_prodigys_scale_only_from_one_hundredth_up(
    lr: float, config: dict[str, Any], expected: float
) -> None:
    _, used = build_prodigy(_params(), config, lr=lr, weight_decay=0.0)

    assert used["lr"] == expected


def test_the_real_prodigy_builds_from_a_config() -> None:
    optimizer, used = build_prodigy(_params(), {"prodigy_beta3": None, "prodigy_decouple": False}, 1e-4, 0.01)

    assert type(optimizer).__name__ == "Prodigy"
    assert used["decouple"] is False and used["weight_decay"] == 0.01
    assert "beta3" not in used


def test_a_setting_the_installed_release_lacks_is_logged_as_ignored(
    fake_prodigy: type[FakeProdigy], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="maltrainer.optimizers.prodigy"):
        build_prodigy(_params(), {"prodigy_slice_p": 4, "prodigy_beta3": 0.9}, lr=1e-4, weight_decay=0.0)

    assert "does not accept" in caplog.text
    assert "slice_p" in caplog.text
    assert "beta3" not in caplog.text
    assert fake_prodigy.built["lr"] == 1.0


def test_the_defaults_a_trainer_passes_replace_the_built_in_ones_and_a_given_key_still_wins(
    fake_prodigy: type[FakeProdigy],
) -> None:
    defaults = {"prodigy_beta2": 0.99, "prodigy_slice_p": None}

    _, used = build_prodigy(_params(), {"adam_beta1": 0.8}, lr=1e-4, weight_decay=0.0, defaults=defaults)
    _, given = build_prodigy(_params(), {"prodigy_beta2": 0.9}, lr=1e-4, weight_decay=0.0, defaults=defaults)

    assert used["betas"] == (0.8, 0.99)
    assert given["betas"] == (0.9, 0.9)
    assert fake_prodigy.built["betas"] == (0.9, 0.9)


def test_a_none_default_drops_the_setting_unless_the_config_gives_it() -> None:
    _, without = build_prodigy(_params(), {}, lr=1.0, weight_decay=0.0, defaults={"prodigy_slice_p": None})
    _, given = build_prodigy(
        _params(), {"prodigy_slice_p": 4}, lr=1.0, weight_decay=0.0, defaults={"prodigy_slice_p": None}
    )

    assert "slice_p" not in without
    assert given["slice_p"] == 4
