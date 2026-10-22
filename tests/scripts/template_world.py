"""A fake trainer subprocess, a datasets and models folder, and the rendered config for train_from_template."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import train_from_template as tft

ROOT = tft.ROOT


class Launch:
    """The trainer subprocess: records its command line and working directory, exits with ``code``."""

    def __init__(self) -> None:
        self.commands: list[tuple[list[str], Path]] = []
        self.code = 0

    def call(self, command: list[str], cwd: Path) -> int:
        self.commands.append((command, cwd))
        return self.code


def install_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Launch:
    """Run from an empty folder (the templates write their outputs relative to it) with a fake subprocess."""
    launched = Launch()
    monkeypatch.setattr(tft, "subprocess", SimpleNamespace(call=launched.call))
    monkeypatch.chdir(tmp_path)
    for name in ("COMFYUI_MODELS_PATH", "DATASETS_PATH"):
        monkeypatch.delenv(name, raising=False)
    return launched


def make_world(tmp_path: Path, model: str = "krea2", dataset: str = "alice") -> tuple[Path, Path]:
    """A datasets folder holding ``dataset`` and a models folder holding the model's files, one of them nested."""
    datasets, models = tmp_path / "datasets", tmp_path / "models"
    (datasets / dataset).mkdir(parents=True)
    for index, (subfolder, name) in enumerate(tft.MODELS[model][3].values()):
        folder = models / subfolder / ("nested" if index == 1 else "")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).touch()
    return datasets, models


def run(monkeypatch: pytest.MonkeyPatch, *argv: object) -> int:
    monkeypatch.setattr("sys.argv", ["train_from_template.py", *map(str, argv)])
    return tft.main()


def rendered_config(tmp_path: Path, model: str = "krea2", dataset: str = "alice") -> dict[str, Any]:
    text = (tmp_path / "output" / model / dataset / "config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)
