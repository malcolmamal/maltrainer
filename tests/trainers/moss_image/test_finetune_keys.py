"""Key commands while a finetune trains: save, quit, pause, the preview menu, the gallery and help."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import train_moss_image_finetune as finetune
import train_moss_image_lora as lora

from .fakes import capture_menu, script_keys
from .runs import run_finetune, same_values
from .world import SAMPLE, Workspace, named_weights


@pytest.fixture
def press(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[list[str]]], None]:
    """``press([[], ['s']])``: the keys pending at the 1st and the 2nd step boundary."""

    def pressing(script: list[list[str]]) -> None:
        script_keys(monkeypatch, [finetune, lora], script)

    return pressing


def _saved(ws: Workspace) -> list[str]:
    return sorted(path.name for path in ws.out.glob("*.safetensors"))


def test_s_saves_a_numbered_checkpoint_with_its_state_and_training_continues(
    workspace: Workspace, press: Callable[..., None]
) -> None:
    workspace.add_images("p", 2)
    press([[], ["s"]])

    trainer = run_finetune(workspace.config(train={"max_steps": 3}))

    assert _saved(workspace) == ["ft.safetensors", "ft_step_2.safetensors"]
    assert trainer.global_step == 3


def test_q_saves_and_stops_without_a_final_file_leaving_a_state_to_resume_from(
    workspace: Workspace, press: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    press([[], ["q"]])

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(workspace.config(train={"max_steps": 5}))

    assert trainer.global_step == 2 and _saved(workspace) == ["ft_step_2.safetensors"]
    assert (workspace.out / "ft_state" / "trainer_state.pt").exists()
    assert "MOSS-Image finetuning stopped by key at step 2; resume from the _state folder." in caplog.text


def test_q_does_not_stop_on_a_save_that_failed(
    workspace: Workspace, press: Callable[..., None], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    real, calls = finetune.save_full_model, []

    def flaky(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        real(*args, **kwargs)

    monkeypatch.setattr(finetune, "save_full_model", flaky)
    press([["q"]])

    with caplog.at_level(logging.ERROR):
        trainer = run_finetune(workspace.config(train={"max_steps": 3}))

    assert trainer.global_step == 3 and _saved(workspace) == ["ft.safetensors"]
    assert "Key [q]: step 1 was not saved, so training continues" in caplog.text


def test_p_pauses_until_p_again_and_h_lists_the_keys(
    workspace: Workspace, press: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    press([["p"], ["h", "p"]])

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(workspace.config(train={"max_steps": 2}))

    assert trainer.global_step == 2
    assert "Paused at step 1: [p] resume" in caplog.text and "Resumed at step 1 after" in caplog.text
    assert "[s] save a checkpoint now" in caplog.text


def test_e_opens_the_preview_menu_and_enter_applies_its_changes(
    workspace: Workspace, press: Callable[..., None], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    typed = iter(["right", "space", "enter"])
    monkeypatch.setattr(lora, "read_key", lambda: next(typed))
    screen = capture_menu(monkeypatch, lora)
    press([["e"]])

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(workspace.config(sample={**SAMPLE, "every": 50}, train={"max_steps": 2}))

    assert trainer.previews.every == 150 and "Preview settings" in screen.getvalue()
    assert "[sample] preview settings changed" in caplog.text and "every: 50 -> 150" in caplog.text


def test_a_menu_that_fails_is_logged_and_training_goes_on(
    workspace: Workspace, press: Callable[..., None], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)

    def broken() -> str:
        raise RuntimeError("no console")

    monkeypatch.setattr(lora, "read_key", broken)
    capture_menu(monkeypatch, lora)
    press([["e"]])

    with caplog.at_level(logging.ERROR):
        trainer = run_finetune(workspace.config(sample={**SAMPLE, "every": 50}, train={"max_steps": 2}))

    assert trainer.global_step == 2 and "The menu failed; training continues" in caplog.text


def _gallery_files(ws: Workspace) -> list[str]:
    return sorted(path.name for path in (ws.out / "samples" / "gallery").iterdir())


def test_g_renders_the_gallery_with_the_average_and_hands_the_raw_weights_back(
    workspace: Workspace, press: Callable[..., None]
) -> None:
    workspace.add_images("p", 2)
    ema = {"ema_decay": 0.5, "ema_warmup": False, "max_steps": 2}
    quiet = run_finetune(workspace.config(output={"path": str(workspace.root / "quiet")}, train=ema))
    press([["g"]])

    shown = run_finetune(workspace.config(sample={**SAMPLE, "every": 100}, train=ema))

    assert _gallery_files(workspace) == ["step000001.png", "step000001.txt", "step000001_zed.png"]
    text = Path(workspace.out / "samples" / "gallery" / "step000001.txt").read_text(encoding="utf-8")
    assert text.startswith("zed  seed ") and "photo of zed, a person" in text
    assert same_values(named_weights(shown.transformer), named_weights(quiet.transformer))
    assert shown.transformer.training


def test_g_is_not_offered_when_the_gallery_is_switched_off(workspace: Workspace, press: Callable[..., None]) -> None:
    workspace.add_images("p", 1)
    press([["g"]])

    run_finetune(workspace.config(sample={**SAMPLE, "every": 100, "gallery": False}, train={"max_steps": 1}))

    assert not (workspace.out / "samples" / "gallery").exists()
