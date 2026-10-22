"""The progress line the web UI parses out of training.log, and the dry run's simulated steps."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from maltrainer.training.progress import log_step, simulate_steps

LOGGER = logging.getLogger("test_progress")


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _automagic(wrapped: bool = False) -> Any:
    groups = [{"name": "blocks", "lr": 0.5}, {"lr": 0.5}]  # the second group has no name: it is listed by index
    inner = SimpleNamespace(
        param_groups=groups,
        get_avg_learning_rate=lambda: 1.5e-4,
        get_learning_rates=lambda: [1e-4, 2e-5],
    )
    return SimpleNamespace(param_groups=groups, optimizer=inner) if wrapped else inner


def test_a_plain_optimizer_logs_the_first_groups_rate_with_the_one_based_epoch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-2}, {"lr": 5.0}])

    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_step(LOGGER, optimizer, 5, 20, torch.tensor(0.5), epoch=1, num_epochs=3)

    assert _lines(caplog) == ["Step: 5/20 [25%], Loss: 0.500000, Epoch: 2/3, LR: 1.00e-02"]


def test_the_note_precedes_the_rate_and_the_suffix_ends_the_line(caplog: pytest.LogCaptureFixture) -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-2}])

    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_step(LOGGER, optimizer, 1, 3, torch.tensor(1.0), 0, 1, note="Reg: 1/2, ", suffix=" [end]")

    assert _lines(caplog) == ["Step: 1/3 [33%], Loss: 1.000000, Epoch: 1/1, Reg: 1/2, LR: 1.00e-02 [end]"]


def test_prodigy_logs_the_multiplier_times_d(caplog: pytest.LogCaptureFixture) -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.5, "d": 4e-3}])

    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_step(LOGGER, optimizer, 2, 4, torch.tensor(0.0), 0, 1)

    assert _lines(caplog) == ["Step: 2/4 [50%], Loss: 0.000000, Epoch: 1/1, LR: 2.00e-03 (mult 0.500 x d 4.00e-03)"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_automagic3_logs_the_average_and_each_group_by_name_or_index(
    caplog: pytest.LogCaptureFixture, wrapped: bool
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_step(LOGGER, _automagic(wrapped), 1, 2, torch.tensor(2.0), 0, 1)

    assert _lines(caplog) == [
        "Step: 1/2 [50%], Loss: 2.000000, Epoch: 1/1, LR: 1.50e-04 (automagic3: blocks 1.00e-04, 1 2.00e-05)"
    ]


def test_automagic3_without_per_group_rates_logs_the_average_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_step(LOGGER, _automagic(), 1, 2, torch.tensor(2.0), 0, 1, note="n ", rates=False, suffix="!")

    assert _lines(caplog) == ["Step: 1/2 [50%], Loss: 2.000000, Epoch: 1/1, n LR: 1.50e-04 (automagic3)!"]


def test_the_dry_run_ticks_every_step_and_says_where_checkpoints_would_land(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        simulate_steps(LOGGER, (2, 3, 6), 0.0, show=False, save_every=2)

    steps = [line for line in _lines(caplog) if line.startswith("Step")]
    assert steps[0] == "Step: 1/6 [16%], Loss: 0.000000, Epoch: 1/3, LR: 0.00e+00 (dry run)"
    assert [line.split("Epoch: ")[1][:3] for line in steps] == ["1/3", "1/3", "2/3", "2/3", "3/3", "3/3"]
    assert [line for line in _lines(caplog) if "would save" in line] == [
        "Dry run: would save checkpoint at step 2",
        "Dry run: would save checkpoint at step 4",
        "Dry run complete: would save final checkpoint at step 6",
    ]


def test_the_dry_run_shows_its_bar_only_when_asked(capsys: pytest.CaptureFixture[str]) -> None:
    simulate_steps(LOGGER, (1, 1, 3), 0.0, show=False)
    hidden = capsys.readouterr().err
    simulate_steps(LOGGER, (1, 1, 3), 0.0, show=True)
    shown = capsys.readouterr().err

    assert hidden == ""
    assert "Training Steps (dry run):" in shown  # tqdm prints the description, then a colon
    assert "3/3" in shown
    assert "4/3" not in shown
    assert "6/3" not in shown
