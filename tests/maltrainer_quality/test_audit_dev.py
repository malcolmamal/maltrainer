"""The dev audit exports the ``dev`` group from uv.lock and audits that file, pinned and without pip."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from maltrainer_quality import audit_dev


def test_the_export_asks_uv_for_the_dev_group_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(command=command, **options)
        return subprocess.CompletedProcess(command, 0, stdout=b"ruff==0.15.18\n")

    monkeypatch.setattr(audit_dev.subprocess, "run", fake_run)

    assert audit_dev.exported(tmp_path) == "ruff==0.15.18\n"
    assert seen["cwd"] == tmp_path and seen["check"] is True
    assert seen["command"][2:] == ["uv", *audit_dev.EXPORT]
    assert "--only-group" in seen["command"] and "dev" in seen["command"]


@pytest.mark.parametrize("code", [0, 1])
def test_the_audit_runs_on_the_exported_pins_and_returns_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    audited: list[tuple[list[str], str]] = []
    monkeypatch.setattr(audit_dev, "exported", lambda _root: "ruff==0.15.18\n")

    def fake_run(command: list[str], **_options: Any) -> subprocess.CompletedProcess[bytes]:
        audited.append((command, Path(command[command.index("-r") + 1]).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(audit_dev.subprocess, "run", fake_run)

    assert audit_dev.main([str(tmp_path)]) == code
    command, pins = audited[0]
    assert command[:3] == [sys.executable, "-m", "pip_audit"]
    assert command[-2:] == ["--no-deps", "--disable-pip"]
    assert pins == "ruff==0.15.18\n"
    assert not Path(command[command.index("-r") + 1]).exists()
