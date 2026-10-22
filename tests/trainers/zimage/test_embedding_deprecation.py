"""The Z-Image textual-inversion trainer is deprecated: starting it says so, the LoRA trainer does not."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest
import yaml

import train_zimage_embedding as embedding
import train_zimage_lora as lora

from .world import Workspace


def _start(monkeypatch: pytest.MonkeyPatch, workspace: Workspace, module: Any, config: dict[str, Any]) -> None:
    path = workspace.root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--dry-run"])
    monkeypatch.setattr(module, "setup_logging", lambda _section: None)
    module.main()


def test_starting_the_embedding_trainer_warns_that_it_is_deprecated(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _start(monkeypatch, workspace, embedding, workspace.embedding_config())

    assert "deprecated" in caplog.text
    assert "proof of concept" in caplog.text


def test_the_lora_trainer_does_not_warn(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _start(monkeypatch, workspace, lora, workspace.lora_config())

    assert "deprecated" not in caplog.text
