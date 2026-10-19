from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maltrainer.utils.previews import Previews


def test_prompts_are_read_from_lines_and_entries_and_blanks_are_dropped(tmp_path: Path) -> None:
    config = {"prompts": ["a cat", "  ", {"prompt": " a dog ", "enabled": False}, {"prompt": ""}, {"enabled": True}]}

    previews = Previews(config, tmp_path)

    assert previews.prompts == ["a cat", "a dog"]
    assert previews.active == ["a cat"]
    assert previews.slots == {"a cat": 0, "a dog": 1}


def test_a_missing_section_turns_previews_off_with_the_defaults(tmp_path: Path) -> None:
    previews = Previews(None, tmp_path / "out")

    assert previews.enabled is False
    assert previews.dir == tmp_path / "out" / "samples"
    assert (previews.every, previews.at_start, previews.negative, previews.steps, previews.cfg) == (
        0,
        True,
        "",
        30,
        3.0,
    )
    assert (previews.width, previews.height, previews.seed, previews.slow_after) == (1024, 1024, 42, 60.0)
    assert previews.rows == {"raw": True, "ema": True}
    assert previews.captions == []


def test_an_interval_without_an_enabled_prompt_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.utils.previews"):
        previews = Previews({"every": 100, "prompts": [{"prompt": "a cat", "enabled": False}]}, tmp_path)

    assert previews.enabled is False
    assert caplog.messages == ["sample.every is set but no sample prompt is enabled: no previews"]


@pytest.mark.parametrize(
    ("rows", "expected"), [(["EMA"], {"raw": False, "ema": True}), (None, {"raw": False, "ema": False})]
)
def test_rows_are_picked_by_name(tmp_path: Path, rows: list | None, expected: dict) -> None:
    assert Previews({"rows": rows}, tmp_path).rows == expected


def test_the_negative_is_encoded_only_with_cfg(tmp_path: Path) -> None:
    with_cfg = Previews({"prompts": ["a", {"prompt": "b", "enabled": False}], "negative": "blurry"}, tmp_path)
    without_cfg = Previews({"prompts": ["a"], "negative": "blurry", "cfg": 1}, tmp_path)

    assert with_cfg.captions == ["a", "b", "blurry"]
    assert without_cfg.captions == ["a"]


@pytest.mark.parametrize(
    ("step", "at_start", "expected"),
    [(0, True, True), (0, False, False), (300, True, True), (301, True, False), (450, True, True), (600, True, True)],
)
def test_previews_are_due_at_the_start_every_interval_and_at_the_end(
    tmp_path: Path, step: int, at_start: bool, expected: bool
) -> None:
    previews = Previews({"every": 300, "at_start": at_start, "prompts": ["a"]}, tmp_path)

    assert previews.due(step, total=450) is expected


def test_nothing_is_due_while_previews_are_off(tmp_path: Path) -> None:
    previews = Previews({"every": 300, "prompts": ["a"]}, tmp_path)
    previews.set_every(-5)

    assert previews.every == 0
    assert previews.due(0, total=450) is False


def test_an_update_lists_every_change_and_keeps_the_slots(tmp_path: Path) -> None:
    previews = Previews({"every": 300, "prompts": ["a", {"prompt": "b", "enabled": False}, "d"]}, tmp_path)

    changes = previews.update(
        {"every": 100, "rows": ["raw"], "prompts": [{"prompt": "b"}, "c", {"prompt": "e", "enabled": False}, "d"]}
    )

    assert changes == [
        "every: 300 -> 100",
        "rows: ema off",
        "enabled  [01] b",
        "+ [03] c",
        "+ [04] e (disabled)",
        "- [00] a",
    ]
    assert previews.slots == {"a": 0, "b": 1, "d": 2, "c": 3, "e": 4}


def test_a_disabled_prompt_is_reported_and_a_returning_prompt_keeps_its_slot(tmp_path: Path) -> None:
    previews = Previews({"prompts": ["a", "b"]}, tmp_path)
    previews.update({"prompts": ["b"]})

    changes = previews.update({"prompts": [{"prompt": "b", "enabled": False}, "a"]})

    assert changes == ["disabled [01] b", "+ [00] a"]


def test_an_update_writes_the_listing_with_removed_prompts(tmp_path: Path) -> None:
    previews = Previews({"every": 300, "prompts": ["a", "b"], "seed": 10}, tmp_path)

    previews.update({"every": 300, "prompts": [{"prompt": "b", "enabled": False}], "seed": 10, "cfg": 4.5, "rows": []})

    assert (tmp_path / "samples" / "prompts.txt").read_text(encoding="utf-8").splitlines() == [
        "00  seed 10  [removed]  a",
        "01  seed 11  [off]  b",
        "every 300 steps, rows none; negative: '', 30 steps, CFG 4.5, 1024x1024",
    ]


def test_an_unchanged_update_reports_nothing_and_writes_nothing(tmp_path: Path) -> None:
    config = {"every": 300, "prompts": ["a"], "rows": ["ema", "raw"]}
    previews = Previews(config, tmp_path)

    assert previews.update(config) == []
    assert not (tmp_path / "samples").exists()


def test_prompts_and_rows_can_be_switched_directly(tmp_path: Path) -> None:
    previews = Previews({"every": 300, "prompts": ["a", "b"]}, tmp_path)

    previews.set_enabled("a", False)
    previews.set_enabled("a", False)
    previews.set_row("ema", False)

    assert previews.active == ["b"]
    assert previews.rows == {"raw": True, "ema": False}
    previews.set_enabled("a", True)
    assert previews.active == ["a", "b"]
