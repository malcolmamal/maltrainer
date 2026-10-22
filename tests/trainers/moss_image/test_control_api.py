"""The control API against the real trainers: a client pauses the run at its first step, looks, changes the preview
settings, saves, renders the gallery, resumes and stops it, all over HTTP, while the MOSS-Image trainer trains."""

from __future__ import annotations

import logging

import pytest

import train_moss_image_lora as lora
from maltrainer.trainers.moss_image import finetune_loop

from .control_driver import CONTROL, Driver
from .runs import run_finetune, run_lora
from .world import SAMPLE, Workspace


def _steer_a_finetune(d: Driver) -> None:
    d.seen["paused"] = d.call("GET", "/status")
    d.seen["settings"] = d.call("GET", "/preview-settings")
    edit = d.call(
        "PATCH",
        "/preview-settings",
        {"every": 7, "prompts": [{"slot": 0, "enabled": False}, {"slot": 1, "enabled": True}]},
    )
    d.seen["edit"] = d.settled(edit)
    d.seen["save"] = d.settled(d.call("POST", "/commands/save"))
    d.seen["gallery"] = d.settled(d.call("POST", "/commands/gallery"))
    d.seen["settings_after"] = d.call("GET", "/preview-settings")
    d.seen["resume"] = d.settled(d.call("POST", "/commands/resume"))
    d.until(lambda: d.call("GET", "/status")["step"] >= 3, "step 3")
    d.seen["running"] = d.call("GET", "/status")


def test_a_client_steers_a_finetune_over_http_and_it_ends_with_a_resumable_save(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    driver = Driver(_steer_a_finetune)
    monkeypatch.setattr(finetune_loop, "make_preview_host_keys", driver.wrap(finetune_loop.make_preview_host_keys))

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(
            workspace.config(
                sample={**SAMPLE, "every": 50}, train={"max_steps": 1000}, control_api=CONTROL["control_api"]
            )
        )
    driver.finish()

    seen = driver.seen
    assert (seen["paused"]["state"], seen["paused"]["step"]) == ("paused", 1)
    assert seen["paused"]["capabilities"] == {"gallery": True, "preview_settings": True}
    assert seen["settings"]["every"] == 50
    assert [p["enabled"] for p in seen["settings"]["prompts"]] == [True, False]
    assert (seen["edit"]["state"], seen["save"]["state"], seen["gallery"]["state"]) == ("done",) * 3
    assert (seen["resume"]["state"], seen["running"]["state"]) == ("done", "running")
    assert seen["settings_after"]["every"] == 7
    assert [p["enabled"] for p in seen["settings_after"]["prompts"]] == [False, True]
    assert trainer.previews.every == 7 and trainer.previews.active == ["a photo of a person"]
    assert trainer.global_step >= 3
    saved = sorted(path.name for path in workspace.out.glob("*.safetensors"))
    assert "ft_step_1.safetensors" in saved and f"ft_step_{trainer.global_step}.safetensors" in saved
    assert (workspace.out / "ft_state" / "trainer_state.pt").exists()
    assert "MOSS-Image finetuning stopped by key" in caplog.text
    assert "Control API: http://127.0.0.1:" in caplog.text


def _steer_a_lora(d: Driver) -> None:
    d.seen["paused"] = d.call("GET", "/status")
    d.seen["capabilities"] = d.seen["paused"]["capabilities"]
    d.seen["resume"] = d.settled(d.call("POST", "/commands/resume"))
    d.until(lambda: d.call("GET", "/status")["step"] >= 2, "step 2")
    d.seen["gallery"] = d.call("POST", "/commands/gallery")


def test_a_client_steers_a_lora_too_and_the_gallery_it_lacks_is_a_409_not_a_crash(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 2)
    driver = Driver(_steer_a_lora)
    monkeypatch.setattr(lora, "make_preview_host_keys", driver.wrap(lora.make_preview_host_keys))

    trainer = run_lora(
        workspace.lora_config(
            sample={**SAMPLE, "every": 50}, train={"max_steps": 1000}, control_api=CONTROL["control_api"]
        )
    )
    driver.finish()

    assert (driver.seen["paused"]["state"], driver.seen["paused"]["step"]) == ("paused", 1)
    assert driver.seen["capabilities"] == {"gallery": False, "preview_settings": True}
    assert driver.seen["resume"]["state"] == "done"
    assert driver.seen["gallery"] == {"detail": "this trainer has no gallery"}
    assert trainer.previews.every == 50
