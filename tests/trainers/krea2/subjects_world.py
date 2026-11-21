"""Helpers for the multi-subject LoRA tests: subject folders, strangers and the side-by-side options."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image

from .world import Workspace, write_image

PLAIN_SGD: dict[str, Any] = {"optimizer": "adamw", "learning_rate": 1e-3}


def subject(ws: Workspace, name: str, trigger: str, caption: str, count: int, **extra: Any) -> dict[str, Any]:
    for i in range(count):
        write_image(ws.root / name / f"{name}{i}.png", caption)
    return {"path": str(ws.root / name), "trigger": trigger, **extra}


def two(ws: Workspace, first: int = 1, second: int = 3, **isolation: Any) -> dict[str, Any]:
    subjects = [
        subject(ws, "a", "zwxem", "zwxem, a woman smiling", first),
        subject(ws, "b", "qrxt", "qrxt, a man standing", second),
    ]
    return {"subjects": subjects, **isolation}


def strangers(ws: Workspace) -> str:
    folder = ws.root / "strangers"
    folder.mkdir()
    for i in range(2):
        Image.new("RGB", (24, 40), (0, 90, i * 100)).save(folder / f"s{i}.png")
    rows = [{"file": f"s{i}.png", "description": f"stranger {i}"} for i in range(2)]
    (folder / "strangers.json").write_text(json.dumps(rows), encoding="utf-8")
    return str(folder)


def sbs(ws: Workspace, **extra: Any) -> dict[str, Any]:
    return {"enabled": True, "strangers_path": strangers(ws), "per_subject": 2, "panel": 32, **extra}
