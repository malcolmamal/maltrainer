"""What the control API and the training loop share: a queue of commands going in, snapshots going out.

The API thread only ever touches the hub; the training thread takes the commands at a step boundary (the same place
the console keys are read) and publishes a fresh snapshot there. Nothing here knows FastAPI, a trainer or a key."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    SAVING = "saving"
    GALLERY = "gallery"
    STOPPING = "stopping"
    FINISHED = "finished"


class CommandState(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class Command(str, Enum):
    SAVE = "save"
    SAVE_AND_STOP = "save-and-stop"
    PAUSE = "pause"
    RESUME = "resume"
    GALLERY = "gallery"
    PREVIEW_SETTINGS = "preview-settings"


KEY_OF = {Command.SAVE: "s", Command.SAVE_AND_STOP: "q", Command.GALLERY: "g"}
NO_CHANGE = "no change needed"
MAX_PENDING = 50  # commands waiting for the next step boundary


class CommandRefusedError(Exception):
    """The run cannot take this command now (it is stopping, or the trainer lacks the feature)."""


@dataclass(frozen=True)
class Capabilities:
    gallery: bool = False
    preview_settings: bool = False


@dataclass
class CommandRecord:
    id: int
    command: Command
    payload: dict[str, Any] | None = None
    state: CommandState = CommandState.PENDING
    detail: str = ""
    submitted_at: float = field(default_factory=time.time)
    settled_at: float | None = None


@dataclass(frozen=True)
class Pending:
    """A taken command and the console key that carries it out (None: the hub's owner applies it itself)."""

    record: CommandRecord
    key: str | None


class ControlHub:
    def __init__(self, capabilities: Capabilities | None = None, history: int = 200) -> None:
        self.capabilities = capabilities or Capabilities()
        self.state = RunState.RUNNING
        self.stats: dict[str, Any] = {}
        self.settings: dict[str, Any] | None = None
        self.published_at: float | None = None
        self._history = history
        self._lock = threading.Lock()
        self._records: OrderedDict[int, CommandRecord] = OrderedDict()
        self._queue: list[CommandRecord] = []
        self._next_id = 1

    def submit(self, command: Command, payload: dict[str, Any] | None = None) -> CommandRecord:
        """Queue a command; raises CommandRefusedError when it cannot be carried out."""
        with self._lock:
            self._check(command)
            record = CommandRecord(self._next_id, command, payload)
            self._next_id += 1
            self._records[record.id] = record
            while len(self._records) > self._history:
                self._records.popitem(last=False)
            self._queue.append(record)
            return record

    def _check(self, command: Command) -> None:
        if self.state in (RunState.STOPPING, RunState.FINISHED):
            raise CommandRefusedError(f"the run is {self.state.value}")
        if command is Command.GALLERY and not self.capabilities.gallery:
            raise CommandRefusedError("this trainer has no gallery")
        if command is Command.PREVIEW_SETTINGS and not self.capabilities.preview_settings:
            raise CommandRefusedError("this trainer has no preview settings")
        if command is Command.SAVE_AND_STOP and any(r.command is command for r in self._queue):
            raise CommandRefusedError("a save-and-stop is already waiting")
        if len(self._queue) >= MAX_PENDING:
            raise CommandRefusedError(f"{MAX_PENDING} commands are already waiting for the next step")

    def get(self, command_id: int) -> CommandRecord | None:
        with self._lock:
            return self._records.get(command_id)

    def recent(self, limit: int = 50) -> list[CommandRecord]:
        with self._lock:
            return list(self._records.values())[-limit:][::-1]

    def take(self, paused: bool) -> list[Pending]:
        """The queued commands for the loop to act on, in order. Pause and resume are folded into at most one
        toggle: a pause when paused, or a pause then a resume, changes nothing and settles at once."""
        with self._lock:
            queue, self._queue = self._queue, []
        toggles = [r for r in queue if r.command in (Command.PAUSE, Command.RESUME)]
        taken = [Pending(r, KEY_OF.get(r.command)) for r in queue if r not in toggles]
        if toggles:
            want = toggles[-1].command is Command.PAUSE
            for record in toggles if want == paused else toggles[:-1]:
                self.settle(record, True, NO_CHANGE if want == paused else "superseded by a later command")
            if want != paused:
                taken.append(Pending(toggles[-1], "p"))
        return taken

    def settle(self, record: CommandRecord, ok: bool, detail: str = "") -> None:
        with self._lock:
            record.state = CommandState.DONE if ok else CommandState.FAILED
            record.detail = detail
            record.settled_at = time.time()

    def finish(self) -> None:
        """The run is over: no command is taken any more, and those still waiting fail, so none stays pending."""
        with self._lock:
            self.state = RunState.FINISHED
            queue, self._queue = self._queue, []
            for record in queue:
                record.state, record.settled_at = CommandState.FAILED, time.time()
                record.detail = "the run ended before the next step boundary"

    def publish(self, stats: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> None:
        """The training thread's latest numbers and (when it has preview settings) their current values."""
        with self._lock:
            if stats is not None:
                self.stats = stats
            if settings is not None:
                self.settings = settings
            self.published_at = time.time()
