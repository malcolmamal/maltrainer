from __future__ import annotations

import pytest

from maltrainer.control.hub import (
    MAX_PENDING,
    NO_CHANGE,
    Capabilities,
    Command,
    CommandRefusedError,
    CommandState,
    ControlHub,
    RunState,
)


def test_commands_are_taken_in_order_with_the_key_that_carries_them_out() -> None:
    hub = ControlHub(Capabilities(gallery=True))
    for command in (Command.SAVE, Command.GALLERY, Command.SAVE_AND_STOP):
        hub.submit(command)

    taken = hub.take(paused=False)

    assert [(p.record.command, p.key) for p in taken] == [
        (Command.SAVE, "s"),
        (Command.GALLERY, "g"),
        (Command.SAVE_AND_STOP, "q"),
    ]
    assert hub.take(paused=False) == []


def test_a_pause_while_running_becomes_one_p_key_and_stays_pending_until_settled() -> None:
    hub = ControlHub()
    record = hub.submit(Command.PAUSE)

    taken = hub.take(paused=False)

    assert [(p.record, p.key) for p in taken] == [(record, "p")]
    assert record.state is CommandState.PENDING


@pytest.mark.parametrize(
    ("command", "paused"),
    [(Command.PAUSE, True), (Command.RESUME, False)],
)
def test_a_pause_or_resume_that_changes_nothing_settles_at_once(command: Command, paused: bool) -> None:
    hub = ControlHub()
    record = hub.submit(command)

    assert hub.take(paused=paused) == []
    assert (record.state, record.detail) == (CommandState.DONE, NO_CHANGE)


def test_a_pause_then_a_resume_in_one_look_cancel_out() -> None:
    hub = ControlHub()
    pause, resume = hub.submit(Command.PAUSE), hub.submit(Command.RESUME)

    assert hub.take(paused=False) == []
    assert pause.state is resume.state is CommandState.DONE


def test_of_several_toggles_the_last_one_decides_and_carries_the_key() -> None:
    hub = ControlHub()
    first, second, third = hub.submit(Command.PAUSE), hub.submit(Command.RESUME), hub.submit(Command.PAUSE)

    taken = hub.take(paused=False)

    assert [(p.record, p.key) for p in taken] == [(third, "p")]
    assert first.state is second.state is CommandState.DONE
    assert first.detail == "superseded by a later command"


def test_settle_records_the_outcome_and_when() -> None:
    hub = ControlHub()
    record = hub.submit(Command.SAVE)

    hub.settle(record, False, "disk full")

    assert (record.state, record.detail) == (CommandState.FAILED, "disk full")
    assert record.settled_at is not None
    assert hub.get(record.id) is record


@pytest.mark.parametrize("state", [RunState.STOPPING, RunState.FINISHED])
def test_a_run_that_is_stopping_or_finished_takes_no_more_commands(state: RunState) -> None:
    hub = ControlHub()
    hub.state = state

    with pytest.raises(CommandRefusedError, match=state.value):
        hub.submit(Command.SAVE)


def test_commands_a_trainer_lacks_are_refused() -> None:
    hub = ControlHub(Capabilities(gallery=False, preview_settings=False))

    with pytest.raises(CommandRefusedError, match="no gallery"):
        hub.submit(Command.GALLERY)
    with pytest.raises(CommandRefusedError, match="no preview settings"):
        hub.submit(Command.PREVIEW_SETTINGS, {"every": 1})


def test_a_second_save_and_stop_while_one_waits_is_refused() -> None:
    hub = ControlHub()
    hub.submit(Command.SAVE_AND_STOP)

    with pytest.raises(CommandRefusedError, match="already waiting"):
        hub.submit(Command.SAVE_AND_STOP)


def test_only_the_most_recent_commands_are_kept_newest_first() -> None:
    hub = ControlHub(history=3)
    ids = [hub.submit(Command.SAVE).id for _ in range(5)]

    assert [r.id for r in hub.recent()] == [ids[4], ids[3], ids[2]]
    assert hub.get(ids[0]) is None
    assert [r.id for r in hub.recent(limit=2)] == [ids[4], ids[3]]


def test_publish_keeps_what_it_is_not_given_and_stamps_the_time() -> None:
    hub = ControlHub()
    assert hub.published_at is None

    hub.publish(stats={"step": 1})
    hub.publish(settings={"every": 5})

    assert hub.stats == {"step": 1}
    assert hub.settings == {"every": 5}
    assert hub.published_at is not None


def test_finishing_the_run_fails_the_commands_still_waiting_and_takes_no_more() -> None:
    hub = ControlHub()
    waiting = hub.submit(Command.SAVE_AND_STOP)

    hub.finish()

    assert hub.state is RunState.FINISHED
    assert (waiting.state, waiting.detail) == (CommandState.FAILED, "the run ended before the next step boundary")
    assert waiting.settled_at is not None
    assert hub.take(paused=False) == []
    with pytest.raises(CommandRefusedError, match="finished"):
        hub.submit(Command.SAVE)


def test_a_flood_of_commands_is_cut_off_at_a_limit_and_the_queue_drains_when_taken() -> None:
    hub = ControlHub()
    for _ in range(MAX_PENDING):
        hub.submit(Command.SAVE)

    with pytest.raises(CommandRefusedError, match="already waiting for the next step"):
        hub.submit(Command.SAVE)
    assert len(hub.take(paused=False)) == MAX_PENDING
    hub.submit(Command.SAVE)
