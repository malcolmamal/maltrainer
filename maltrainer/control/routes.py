"""The control API's routes, one router per concern, each over a ``ControlHub``."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from maltrainer.control.hub import Command, CommandRefusedError, ControlHub
from maltrainer.control.schemas import (
    Action,
    CapabilitiesOut,
    CommandOut,
    PreviewSettings,
    PreviewSettingsPatch,
    RunStatus,
)


def status_routes(hub: ControlHub) -> APIRouter:
    router = APIRouter()

    @router.get("/status", response_model=RunStatus, summary="Current stats")
    def status() -> RunStatus:
        age = None if hub.published_at is None else round(time.time() - hub.published_at, 1)
        capabilities = CapabilitiesOut(**vars(hub.capabilities))
        return RunStatus(state=hub.state, published_seconds_ago=age, capabilities=capabilities, **hub.stats)

    return router


def submit(hub: ControlHub, command: Command, payload: dict[str, Any] | None = None) -> CommandOut:
    try:
        return CommandOut.of(hub.submit(command, payload))
    except CommandRefusedError as error:
        raise HTTPException(409, str(error)) from error


def command_routes(hub: ControlHub) -> APIRouter:
    router = APIRouter()

    @router.post("/commands/{action}", response_model=CommandOut, status_code=202, summary="Send a command")
    def command(action: Action) -> CommandOut:
        """save [s]: a checkpoint now, training continues. save-and-stop [q]: save, then stop (only once the save
        worked). pause / resume [p]: wait at a step boundary. gallery [g]: render the gallery (training waits)."""
        return submit(hub, Command(action.value))

    @router.get("/commands", response_model=list[CommandOut], summary="Recent commands, newest first")
    def commands() -> list[CommandOut]:
        return [CommandOut.of(r) for r in hub.recent()]

    @router.get("/commands/{command_id}", response_model=CommandOut, summary="One command's state")
    def command_state(command_id: int) -> CommandOut:
        record = hub.get(command_id)
        if record is None:
            raise HTTPException(404, "no such command (only the most recent ones are kept)")
        return CommandOut.of(record)

    return router


def preview_routes(hub: ControlHub) -> APIRouter:
    router = APIRouter()

    def current() -> dict[str, Any]:
        if hub.settings is None:
            raise HTTPException(404, "this trainer has no preview settings")
        return hub.settings

    @router.get("/preview-settings", response_model=PreviewSettings, summary="The preview settings (the `e` menu)")
    def preview_settings() -> dict[str, Any]:
        return current()

    @router.patch("/preview-settings", response_model=CommandOut, status_code=202, summary="Change preview settings")
    def patch_preview_settings(patch: PreviewSettingsPatch) -> CommandOut:
        """Interval, rows and which prompts render, for this run only (the config file is not touched)."""
        settings = current()
        slots = {p["slot"] for p in settings["prompts"]}
        unknown = [t.slot for t in patch.prompts or [] if t.slot not in slots]
        bad_rows = [r for r in patch.rows or {} if r not in settings["rows"]]
        if unknown or bad_rows:
            detail = f"unknown prompt slot(s) {unknown} or row(s) {bad_rows}; see GET /preview-settings"
            raise HTTPException(422, detail)
        return submit(hub, Command.PREVIEW_SETTINGS, patch.model_dump(exclude_none=True))

    return router
