from __future__ import annotations

import threading

from maltrainer.control.hub import Command, CommandState, RunState

from .helpers import Saver
from .helpers import controlled_keys as _keys


def test_pause_waits_at_the_step_until_resume_and_settles_both_commands() -> None:
    keys, hub = _keys()
    pause = hub.submit(Command.PAUSE)
    seen: list[RunState] = []
    resumed: list[int] = []

    def resume_later() -> None:
        seen.append(hub.state)
        resumed.append(hub.submit(Command.RESUME).id)

    threading.Timer(0.3, resume_later).start()
    assert keys.handle(4, Saver()) is False

    assert seen == [RunState.PAUSED]
    record = hub.get(resumed[0])
    assert record is not None
    assert (pause.state, record.state) == (CommandState.DONE, CommandState.DONE)
    assert hub.state is RunState.RUNNING


def test_a_save_command_during_a_pause_is_done_while_still_paused() -> None:
    keys, hub = _keys()
    hub.submit(Command.PAUSE)
    save = Saver()
    during: list[CommandState] = []
    saved = hub.submit(Command.SAVE)

    def check_then_resume() -> None:
        during.append(saved.state)
        hub.submit(Command.RESUME)

    threading.Timer(0.5, check_then_resume).start()
    keys.handle(2, save)

    assert save.steps == [2]
    assert during == [CommandState.DONE]


def test_a_pause_shows_the_step_it_waits_at() -> None:
    keys, hub = _keys()
    hub.submit(Command.PAUSE)
    during: list[object] = []

    def look_then_resume() -> None:
        during.append(hub.stats["step"])
        hub.submit(Command.RESUME)

    threading.Timer(0.3, look_then_resume).start()
    keys.handle(6, Saver())

    assert during == [6]


def test_a_resume_while_running_changes_nothing_and_never_pauses_the_loop() -> None:
    keys, hub = _keys()
    record = hub.submit(Command.RESUME)

    assert keys.handle(1, Saver()) is False
    assert (record.state, record.detail) == (CommandState.DONE, "no change needed")
