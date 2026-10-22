"""``KeyCommands`` that also take the control API's commands, at the same step boundary and through the same code.

Each API command becomes the console key it stands for (save ``s``, save and stop ``q``, pause or resume ``p``, gallery
``g``) among the keys ``poll`` returns, so ``handle`` and the pause loop treat both sources alike, including never
stopping on a save that failed. What differs is the book-keeping: a command is settled (done or failed) after the loop
acted on it, and the hub's snapshot is refreshed at every step."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from maltrainer.control.hub import Command, ControlHub, Pending, RunState
from maltrainer.control.previews_patch import apply_preview_patch, preview_snapshot
from maltrainer.control.stats import RunView
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.previews import Previews

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class PreviewTarget:
    previews: Previews
    row_labels: Callable[[], list[str]]


class ControlledKeys(KeyCommands):
    def __init__(
        self,
        hub: ControlHub,
        view: RunView,
        previews: PreviewTarget | None = None,
        menu: Callable[[], None] | None = None,
        gallery: Callable[[], None] | None = None,
    ) -> None:
        self.hub, self.view, self.target = hub, view, previews
        self.on_close: Callable[[], None] | None = None
        self.url = ""  # where the API listens, once it runs
        self._inner_gallery = gallery
        self._taken: list[Pending] = []
        self._paused = False
        self._publish_failed = False
        self._saved: bool | None = None
        self._gallery_ok: bool | None = None
        super().__init__(menu=menu, gallery=self._run_gallery if gallery is not None else None)
        self.hub.publish(settings=self._settings())

    def _settings(self) -> dict[str, Any] | None:
        return None if self.target is None else preview_snapshot(self.target.previews, self.target.row_labels())

    def _in_state(self, state: RunState, work: Callable[[], T]) -> T:
        before = self.hub.state
        self.hub.state = state
        try:
            return work()
        finally:
            self.hub.state = before

    def _run_gallery(self) -> None:
        self._gallery_ok = False
        if self._inner_gallery is not None:
            self._in_state(RunState.GALLERY, self._inner_gallery)
            self._gallery_ok = True

    def poll(self) -> list[str]:
        self._settle()
        keys = super().poll()
        for pending in self.hub.take(self._paused):
            if pending.record.command is Command.PREVIEW_SETTINGS:
                self._apply_settings(pending)
                continue
            self._taken.append(pending)
            if pending.key is not None:
                keys.append(pending.key)
        return keys

    def _apply_settings(self, pending: Pending) -> None:
        if self.target is None:  # the hub refuses these when the trainer has no previews; belt and braces
            self.hub.settle(pending.record, False, "this trainer has no preview settings")
            return
        try:
            changes = apply_preview_patch(self.target.previews, pending.record.payload or {})
        except Exception as error:  # a bad edit must not end the run; the caller sees it on the command
            logger.exception("The preview settings change failed; training continues")
            self.hub.settle(pending.record, False, str(error))
            return
        self.hub.settle(pending.record, True, "; ".join(changes) or "no change needed")
        self.hub.publish(settings=self._settings())

    def _settle(self) -> None:
        """The commands the loop acted on since the last look: done, or failed when their work did not happen."""
        for pending in self._taken:
            ok, detail = self._outcome(pending.record.command)
            self.hub.settle(pending.record, ok, detail)
        self._taken.clear()
        self._saved = self._gallery_ok = None

    def _outcome(self, command: Command) -> tuple[bool, str]:
        if command in (Command.SAVE, Command.SAVE_AND_STOP):
            if self._saved is False:
                stays = ", training continues" if command is Command.SAVE_AND_STOP else ""
                return False, "the checkpoint was not saved" + stays
            return True, "saved"
        if command is Command.GALLERY:
            return (True, "gallery rendered") if self._gallery_ok else (False, "the gallery failed, see the log")
        return True, "paused" if command is Command.PAUSE else "resumed"

    def _save_tracked(self, save: Callable[[int], bool | None]) -> Callable[[int], bool | None]:
        def tracked(step: int) -> bool | None:
            result: bool | None = self._in_state(RunState.SAVING, lambda: save(step))
            self._saved = result is not False
            return result

        return tracked

    def _pause(self, step: int, save: Callable[[int], bool | None]) -> bool:
        self._paused = True
        try:
            return self._in_state(RunState.PAUSED, lambda: KeyCommands._pause(self, step, save))
        finally:
            self._paused = False

    def handle(self, step: int, save: Callable[[int], bool | None]) -> bool:
        self._publish(step)  # first: a pause at this step must show this step
        stop = super().handle(step, self._save_tracked(save))
        self._settle()
        if stop:
            self.hub.state = RunState.STOPPING
        return stop

    def _publish(self, step: int) -> None:
        """The snapshot for the API. It watches the run and must never end it, so a failure is logged, once."""
        try:
            self.hub.publish(stats=self.view.stats(step), settings=self._settings())
        except Exception:
            if not self._publish_failed:
                logger.exception("The control API could not read the run's stats; training continues")
            self._publish_failed = True

    def close(self) -> None:
        super().close()
        self.hub.finish()
        if self.on_close is not None:
            self.on_close()
