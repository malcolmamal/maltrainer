"""Editing the config file while a LoRA run goes: what the sample: reload applies, reports and ignores."""

from __future__ import annotations

import logging
import os

import pytest

from .watch import wait_for_encoding, watched
from .world import SAMPLE, Workspace


def test_an_unchanged_file_is_left_alone(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    trainer, _ = watched(workspace)

    with caplog.at_level(logging.INFO):
        trainer._watch_config()

    assert "[sample]" not in caplog.text and trainer._encoding is None


def test_edits_outside_the_sample_section_are_reported_and_not_applied(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, rewrite = watched(workspace)
    rewrite({"train": {"learning_rate": 5.0}, "sample": {**SAMPLE, "every": 9}})

    with caplog.at_level(logging.INFO):
        trainer._watch_config()

    assert trainer.config["train"]["learning_rate"] == 1e-3 and trainer.previews.every == 9
    assert "changed outside sample: (train): those apply at the next start" in caplog.text


def test_an_edit_that_leaves_the_sample_section_alone_changes_nothing_but_is_reported(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, rewrite = watched(workspace)
    rewrite({"train": {"learning_rate": 5.0}})

    with caplog.at_level(logging.INFO):
        trainer._watch_config()
        wait_for_encoding(trainer)

    assert "[sample] reloaded" not in caplog.text and "changed outside sample: (train)" in caplog.text
    assert trainer.previews.every == 2


def test_a_half_saved_file_keeps_the_previous_settings_and_says_so(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, rewrite = watched(workspace)
    rewrite({})
    assert trainer.config_path is not None
    trainer.config_path.write_text("sample: [unclosed", encoding="utf-8")
    os.utime(trainer.config_path, ns=(1, 2 * 10**18))

    with caplog.at_level(logging.WARNING):
        trainer._watch_config()

    assert (
        trainer.previews.every == 2
        and "could not be read" in caplog.text
        and "the previous settings stay" in caplog.text
    )


def test_a_second_process_does_not_watch_the_file(workspace: Workspace) -> None:
    trainer, rewrite = watched(workspace)
    rewrite({"sample": {**SAMPLE, "every": 7}})
    trainer.is_main_process = False

    trainer._watch_config()

    assert trainer.previews.every == 2


def test_a_file_that_vanished_is_ignored(workspace: Workspace) -> None:
    trainer, _ = watched(workspace)
    assert trainer.config_path is not None
    trainer.config_path.unlink()

    trainer._watch_config()

    assert trainer.previews.every == 2
