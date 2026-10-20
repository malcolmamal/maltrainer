"""Key commands while a LoRA trains: save, quit, pause and help, acted on at the next step boundary."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from types import SimpleNamespace

import pytest

import train_krea2_lora
from maltrainer.utils import keyboard

from .fakes import ScriptedKeys, script_keys
from .runs import run_lora
from .world import Workspace


@pytest.fixture
def press(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[list[str]]], list[ScriptedKeys]]:
    """``press([[], ['s']])``: the keys pending at the 1st and the 2nd step boundary."""

    def pressing(script: list[list[str]]) -> list[ScriptedKeys]:
        return script_keys(monkeypatch, [train_krea2_lora], script)

    return pressing


def _saved(ws: Workspace) -> list[str]:
    return sorted(path.name for path in ws.out.glob("*.safetensors"))


def test_s_saves_a_numbered_checkpoint_and_training_continues(
    workspace: Workspace, press: Callable[..., list[ScriptedKeys]]
) -> None:
    workspace.add_images(2)
    press([[], ["s"]])

    trainer = run_lora(workspace.lora_config())

    assert _saved(workspace) == ["k2.safetensors", "k2_step_2.safetensors"]
    assert trainer.last_saved_path == str(workspace.out / "k2.safetensors")


def test_q_saves_and_stops_without_a_final_file(
    workspace: Workspace, press: Callable[..., list[ScriptedKeys]], caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    press([[], ["q"]])

    with caplog.at_level(logging.INFO):
        trainer = run_lora(workspace.lora_config(train={"max_steps": 5}))

    assert _saved(workspace) == ["k2_step_2.safetensors"]
    assert trainer.last_saved_path == str(workspace.out / "k2_step_2.safetensors")
    assert "Krea 2 LoRA training stopped by key at step 2 (saved as step 2)." in caplog.text
    assert "Krea 2 LoRA training completed!" not in caplog.text


def test_p_pauses_until_p_again_and_h_lists_the_keys(
    workspace: Workspace,
    press: Callable[..., list[ScriptedKeys]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace.add_images(2)
    made = press([["p"], ["h", "p"]])
    # The scripted pause lasts microseconds, and Windows' clock ticks every ~15 ms: a clock that moves on each
    # reading makes the measured pause certain instead of depending on where the tick falls.
    seconds = itertools.count(1000)
    monkeypatch.setattr(keyboard, "time", SimpleNamespace(time=lambda: float(next(seconds)), sleep=lambda _s: None))

    with caplog.at_level(logging.INFO):
        run_lora(workspace.lora_config(train={"max_steps": 2}))

    assert "Paused at step 1: [p] resume" in caplog.text and "Resumed at step 1 after" in caplog.text
    assert "[s] save a checkpoint now" in caplog.text
    assert made[0].script == [] and made[0].paused_seconds > 0


def test_every_step_polls_the_keys_once_and_the_listener_is_closed_when_the_run_ends(
    workspace: Workspace, press: Callable[..., list[ScriptedKeys]], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    closed: list[bool] = []
    monkeypatch.setattr(ScriptedKeys, "close", lambda _self: closed.append(True))
    made = press([[], [], [], []])

    run_lora(workspace.lora_config())

    assert len(made[0].script) == 1 and closed == [True]
