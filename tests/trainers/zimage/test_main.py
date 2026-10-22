"""``main()`` of both trainers: the command line, the YAML config, logging setup and the saved config copy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_zimage_embedding as embedding
import train_zimage_lora as lora

from .world import Workspace

KINDS = pytest.mark.parametrize("kind", ["lora", "embedding"])


def _module(kind: str) -> Any:
    return lora if kind == "lora" else embedding


def _argv(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *flags: str) -> Path:
    path = ws.root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), *flags])
    return path


def _config(ws: Workspace, kind: str, **sections: Any) -> dict[str, Any]:
    return ws.lora_config(**sections) if kind == "lora" else ws.embedding_config(**sections)


@KINDS
def test_main_trains_from_the_yaml_file_and_keeps_a_copy_of_the_config_with_the_output(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    workspace.add_images(2)
    config = _config(workspace, kind, logging={"level": "DEBUG"})
    _argv(monkeypatch, workspace, config)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_module(kind), "setup_logging", calls.append)

    _module(kind).main()

    assert calls == [{"level": "DEBUG"}]
    assert yaml.safe_load((workspace.out / "config.yaml").read_text(encoding="utf-8")) == config
    assert (workspace.out / f"zimage_{kind}_step_final.safetensors").exists()


@KINDS
def test_main_sets_up_logging_with_an_empty_section_when_the_config_has_none(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _argv(monkeypatch, workspace, _config(workspace, kind), "--dry-run")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_module(kind), "setup_logging", calls.append)

    _module(kind).main()

    assert calls == [{}]


@KINDS
def test_the_dry_run_flags_simulate_the_steps_without_loading_a_model(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str, fakes: Any
) -> None:
    workspace.add_images(2)
    _argv(monkeypatch, workspace, _config(workspace, kind), "--dry-run", "--dry-run-step-time", "0")

    _module(kind).main()

    assert fakes.loads == []
    assert (workspace.out / f"zimage_{kind}_step_final.safetensors").read_text(encoding="utf-8").startswith("{")


@KINDS
def test_a_config_without_an_output_section_is_not_copied_and_writes_to_the_default_folder(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.chdir(workspace.root)
    _argv(monkeypatch, workspace, _config(workspace, kind, output=None), "--dry-run", "--dry-run-step-time", "0")

    _module(kind).main()

    assert not (workspace.root / "output" / "config.yaml").exists()
    assert (workspace.root / "output" / f"zimage_{kind}_step_final.safetensors").exists()


@KINDS
def test_main_requires_the_config_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["train"])

    with pytest.raises(SystemExit) as stop:
        _module(kind).main()

    assert stop.value.code == 2
    assert "the following arguments are required: --config" in capsys.readouterr().err


@KINDS
def test_main_reports_a_config_file_that_does_not_exist(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(workspace.root / "nope.yaml")])

    with pytest.raises(FileNotFoundError, match=r"Config file not found: .*nope\.yaml"):
        _module(kind).main()
