from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maltrainer.control.hub import Command, CommandState, ControlHub, RunState
from maltrainer.control.keys import PreviewTarget
from maltrainer.utils.previews import Previews

from .helpers import Saver
from .helpers import controlled_keys as _keys


def test_a_save_command_saves_once_and_training_continues() -> None:
    keys, hub = _keys()
    record = hub.submit(Command.SAVE)
    save = Saver()

    assert keys.handle(5, save) is False

    assert save.steps == [5]
    assert (record.state, record.detail) == (CommandState.DONE, "saved")
    assert hub.state is RunState.RUNNING


def test_save_and_stop_saves_then_stops_the_run() -> None:
    keys, hub = _keys()
    record = hub.submit(Command.SAVE_AND_STOP)
    save = Saver()

    assert keys.handle(7, save) is True

    assert save.steps == [7]
    assert record.state is CommandState.DONE
    assert hub.state is RunState.STOPPING
    keys.close()
    assert hub.state is RunState.FINISHED


def test_save_and_stop_never_stops_on_a_save_that_did_not_happen() -> None:
    keys, hub = _keys()
    record = hub.submit(Command.SAVE_AND_STOP)

    assert keys.handle(7, Saver(result=False)) is False

    assert record.state is CommandState.FAILED
    assert record.detail == "the checkpoint was not saved, training continues"
    assert hub.state is RunState.RUNNING


def test_a_save_and_a_save_and_stop_together_write_the_step_once() -> None:
    keys, hub = _keys()
    first, second = hub.submit(Command.SAVE), hub.submit(Command.SAVE_AND_STOP)
    save = Saver()

    assert keys.handle(3, save) is True

    assert save.steps == [3]
    assert first.state is second.state is CommandState.DONE


def test_the_state_reads_saving_while_the_checkpoint_is_written() -> None:
    keys, hub = _keys()
    hub.submit(Command.SAVE)
    seen: list[RunState] = []

    def save(_step: int) -> None:
        seen.append(hub.state)

    keys.handle(1, save)

    assert seen == [RunState.SAVING]
    assert hub.state is RunState.RUNNING


def test_the_gallery_command_runs_the_gallery_and_the_state_reads_gallery_meanwhile() -> None:
    seen: list[RunState] = []
    holder: list[ControlHub] = []

    def gallery() -> None:
        seen.append(holder[0].state)

    keys, hub = _keys(gallery=gallery)
    holder.append(hub)
    record = hub.submit(Command.GALLERY)

    keys.handle(1, Saver())

    assert seen == [RunState.GALLERY]
    assert (record.state, record.detail) == (CommandState.DONE, "gallery rendered")
    assert hub.state is RunState.RUNNING


def test_a_gallery_that_raises_fails_its_command_and_training_goes_on() -> None:
    def gallery() -> None:
        raise RuntimeError("out of memory")

    keys, hub = _keys(gallery=gallery)
    record = hub.submit(Command.GALLERY)

    assert keys.handle(1, Saver()) is False

    assert record.state is CommandState.FAILED
    assert hub.state is RunState.RUNNING


def test_preview_settings_are_applied_at_the_step_boundary_and_published(tmp_path: Path) -> None:
    previews = Previews({"every": 100, "prompts": ["a", {"prompt": "b", "enabled": False}]}, tmp_path)
    keys, hub = _keys(previews=PreviewTarget(previews, list))
    assert hub.settings is not None and hub.settings["every"] == 100
    record = hub.submit(Command.PREVIEW_SETTINGS, {"every": 250, "prompts": [{"slot": 1, "enabled": True}]})

    keys.handle(1, Saver())

    assert previews.every == 250
    assert previews.enabled_prompts == {"a", "b"}
    assert record.state is CommandState.DONE
    assert "every: 100 -> 250" in record.detail
    assert hub.settings is not None
    assert hub.settings["every"] == 250
    assert [p["enabled"] for p in hub.settings["prompts"]] == [True, True]


def test_a_preview_edit_that_fails_fails_its_command_and_not_the_run(tmp_path: Path) -> None:
    previews = Previews({"every": 100, "prompts": ["a"]}, tmp_path)
    keys, hub = _keys(previews=PreviewTarget(previews, list))
    record = hub.submit(Command.PREVIEW_SETTINGS, {"prompts": [{"slot": 9, "enabled": True}]})

    assert keys.handle(1, Saver()) is False

    assert record.state is CommandState.FAILED
    assert previews.enabled_prompts == {"a"}


def test_every_step_publishes_fresh_stats() -> None:
    keys, hub = _keys()

    keys.handle(1, Saver())
    keys.handle(2, Saver())

    assert hub.stats["step"] == 2
    assert hub.stats["total_steps"] == 100
    assert hub.published_at is not None


def test_close_ends_the_run_and_runs_the_hook_once() -> None:
    keys, hub = _keys()
    calls: list[int] = []
    keys.on_close = lambda: calls.append(1)

    keys.close()

    assert hub.state is RunState.FINISHED
    assert calls == [1]


def test_closing_the_run_fails_a_command_that_never_got_its_step_boundary() -> None:
    keys, hub = _keys()
    record = hub.submit(Command.SAVE)

    keys.close()

    assert record.state is CommandState.FAILED
    assert hub.state is RunState.FINISHED


def test_the_settings_snapshot_follows_changes_made_by_other_routes(tmp_path: Path) -> None:
    previews = Previews({"every": 100, "prompts": ["a"]}, tmp_path)
    keys, hub = _keys(previews=PreviewTarget(previews, lambda: ["raw"]))

    previews.set_every(30)
    previews.set_row("raw", False)
    keys.handle(1, Saver())

    assert hub.settings is not None
    assert (hub.settings["every"], hub.settings["rows"]) == (30, {"raw": False})


def test_a_failure_reading_the_stats_is_logged_once_and_never_stops_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    keys, hub = _keys()
    keys.view.host.optimizer = object()  # no param_groups: reading its rate raises

    with caplog.at_level(logging.ERROR, logger="maltrainer.control.keys"):
        stops = [keys.handle(1, Saver()), keys.handle(2, Saver())]

    assert stops == [False, False]
    assert caplog.text.count("could not read the run's stats") == 1
    assert hub.stats == {}
