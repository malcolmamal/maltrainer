"""Prompts added to the config file mid-run are encoded on the CPU in the background."""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
import torch

from .fakes import Fakes
from .watch import NEW, wait_for_encoding, watched
from .world import SAMPLE, Workspace


def test_a_new_prompt_in_the_file_is_encoded_on_the_cpu_and_joins_the_previews(
    workspace: Workspace, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, rewrite = watched(workspace)
    rewrite({"sample": {**SAMPLE, "prompts": [*SAMPLE["prompts"], NEW]}})

    with caplog.at_level(logging.INFO):
        trainer._watch_config()
        wait_for_encoding(trainer)

    assert NEW in trainer.previews.prompts and trainer.previews.slots[NEW] == 2
    assert trainer.text_cache[NEW].shape[-1] == 8 and fakes.text_loads == [torch.device("cpu")]
    assert "[sample] reloaded from run.yaml" in caplog.text and f"+ [02] {NEW}" in caplog.text
    assert "[sample] encoding 4 new prompt(s) on the CPU in the background" in caplog.text


def test_the_cpu_encoder_is_loaded_once_for_every_later_prompt(workspace: Workspace, fakes: Fakes) -> None:
    trainer, rewrite = watched(workspace)
    for index in range(2):
        rewrite({"sample": {**SAMPLE, "prompts": [*SAMPLE["prompts"], f"{NEW} {index}"]}})
        trainer._watch_config()
        wait_for_encoding(trainer)

    assert len(fakes.text_loads) == 1 and f"{NEW} 1" in trainer.text_cache


def test_a_failing_background_encode_is_logged_and_does_not_end_the_run(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer, rewrite = watched(workspace)

    def out_of_memory(*_args: Any) -> torch.Tensor:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(fakes.text, "encode", out_of_memory)
    rewrite({"sample": {**SAMPLE, "prompts": [*SAMPLE["prompts"], NEW]}})

    with caplog.at_level(logging.ERROR):
        trainer._watch_config()
        wait_for_encoding(trainer)

    assert "[sample] encoding new prompts failed" in caplog.text and NEW not in trainer.text_cache


def test_nothing_is_started_while_an_earlier_encode_is_still_running(workspace: Workspace, fakes: Fakes) -> None:
    trainer, rewrite = watched(workspace)
    release = threading.Event()
    busy = threading.Thread(target=release.wait, daemon=True)
    busy.start()
    trainer._encoding = busy
    rewrite({"sample": {**SAMPLE, "prompts": [*SAMPLE["prompts"], NEW]}})

    trainer._watch_config()
    release.set()
    busy.join(timeout=10)

    assert trainer._encoding is busy and fakes.text_loads == [] and NEW not in trainer.text_cache


def test_a_prompt_is_ready_once_it_and_the_negative_when_cfg_needs_one_are_encoded(workspace: Workspace) -> None:
    trainer, _ = watched(workspace)
    trainer.text_cache["a photo of zed"] = torch.zeros(2, 8)

    assert not trainer._preview_ready("a photo of zed")

    trainer.text_cache[""] = torch.zeros(2, 8)
    assert trainer._preview_ready("a photo of zed") and not trainer._preview_ready("never encoded")
    trainer.previews.cfg = 1.0
    del trainer.text_cache[""]
    assert trainer._preview_ready("a photo of zed")
