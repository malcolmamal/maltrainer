from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from maltrainer.control.previews_patch import apply_preview_patch, preview_snapshot
from maltrainer.control.stats import RunView, learning_rate
from maltrainer.utils.previews import Previews
from maltrainer.utils.train_monitor import TrainMonitor

from .helpers import make_view


def _optimizer(**group: float) -> SimpleNamespace:
    return SimpleNamespace(param_groups=[{"lr": 0.002, **group}])


def test_the_learning_rate_is_the_plain_rate() -> None:
    assert learning_rate(_optimizer()) == 0.002


def test_a_prodigy_rate_is_the_multiplier_times_d() -> None:
    assert learning_rate(_optimizer(d=0.5)) == pytest.approx(0.001)


def test_an_automagic3_rate_is_the_average_even_behind_accelerates_wrapper() -> None:
    inner = SimpleNamespace(get_avg_learning_rate=lambda: 3e-5, param_groups=[{"lr": 1.0}])
    wrapper = SimpleNamespace(optimizer=inner, param_groups=[{"lr": 1.0}])

    assert learning_rate(wrapper) == 3e-5


def test_stats_give_progress_epoch_rate_and_an_eta_from_the_monitors_seconds_per_step() -> None:
    view = make_view(total=200, epoch=2)
    monitor = view.host.monitor
    monitor.record(0.5, 0.3)
    monitor.record(0.7, 0.9)
    monitor.steps = 4
    monitor.started -= 8

    stats = view.stats(60)

    assert (stats["step"], stats["total_steps"], stats["percent"], stats["epoch"]) == (60, 200, 30.0, 3)
    assert stats["learning_rate"] == pytest.approx(0.01)
    assert stats["window"]["window_loss"] == pytest.approx(0.6)
    seconds = stats["window"]["seconds_per_step"]
    assert seconds == pytest.approx(2.0, abs=0.5)
    assert stats["eta_seconds"] == pytest.approx(seconds * 140, rel=0.01)


def test_a_trainer_without_an_epoch_or_a_monitor_reports_nulls_not_errors() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    host = SimpleNamespace(run_metadata={"planned_total_steps": 10}, optimizer=torch.optim.SGD([parameter], lr=0.1))

    stats = RunView(host).stats(5)

    assert stats["epoch"] is None
    assert stats["window"] is None
    assert stats["eta_seconds"] is None
    assert stats["percent"] == 50.0


def test_the_given_monitor_wins_over_the_hosts() -> None:
    view = make_view()
    own = TrainMonitor(every=5)
    own.record(0.25, 0.1)

    assert RunView(view.host, own).stats(1)["window"]["last_loss"] == 0.25


def _previews(tmp_path: Path) -> Previews:
    return Previews({"every": 100, "prompts": ["a", {"prompt": "b", "enabled": False}]}, tmp_path)


def test_the_snapshot_lists_every_prompt_with_its_stable_slot_and_the_rows_asked_for(tmp_path: Path) -> None:
    assert preview_snapshot(_previews(tmp_path), ["raw", "ema"]) == {
        "every": 100,
        "rows": {"raw": True, "ema": True},
        "prompts": [{"slot": 0, "prompt": "a", "enabled": True}, {"slot": 1, "prompt": "b", "enabled": False}],
    }


def test_a_patch_changes_what_it_names_and_reports_each_change(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    previews = _previews(tmp_path)

    with caplog.at_level(logging.INFO, logger="maltrainer.control.previews_patch"):
        changes = apply_preview_patch(
            previews,
            {
                "every": 40,
                "rows": {"ema": False},
                "prompts": [{"slot": 1, "enabled": True}, {"slot": 0, "enabled": False}],
            },
        )

    assert previews.every == 40
    assert previews.rows["ema"] is False
    assert previews.enabled_prompts == {"b"}
    assert changes == ["every: 100 -> 40", "rows: ema off", "enabled  [01] b", "disabled [00] a"]
    assert "by the control API" in caplog.text
    assert "[off]  a" in (tmp_path / "samples" / "prompts.txt").read_text(encoding="utf-8")


def test_a_patch_that_changes_nothing_reports_nothing_and_writes_nothing(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    changes = apply_preview_patch(previews, {"every": 100, "prompts": [{"slot": 0, "enabled": True}]})

    assert changes == []
    assert not (tmp_path / "samples" / "prompts.txt").exists()


def test_every_zero_turns_the_previews_off(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    apply_preview_patch(previews, {"every": 0})

    assert previews.enabled is False


def test_a_patch_naming_a_missing_slot_changes_nothing_at_all(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    with pytest.raises(ValueError, match="no prompt with slot 9"):
        apply_preview_patch(previews, {"every": 5, "rows": {"ema": False}, "prompts": [{"slot": 9, "enabled": True}]})

    assert previews.every == 100
    assert previews.rows["ema"] is True


def test_stats_without_a_planned_total_report_nulls_not_an_error() -> None:
    view = make_view()
    view.host.run_metadata.clear()

    stats = view.stats(5)

    assert (stats["total_steps"], stats["percent"], stats["eta_seconds"]) == (None, None, None)


def test_a_run_is_a_lora_unless_the_view_says_it_is_a_finetune() -> None:
    lora = make_view()
    finetune = RunView(lora.host, kind="finetune")

    assert lora.stats(1)["kind"] == "lora"
    assert finetune.stats(1)["kind"] == "finetune"
