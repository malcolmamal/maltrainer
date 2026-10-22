"""``exported`` runs uv from the project root and hands back its output with Unix line endings."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from maltrainer_quality import requirements


def test_the_export_runs_uv_in_the_root_and_normalises_line_endings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(command=command, **options)
        return subprocess.CompletedProcess(command, 0, stdout=b"numpy==2.0.0\r\npyyaml==6.0.1\r\n")

    monkeypatch.setattr(requirements.subprocess, "run", fake_run)

    assert requirements.exported(tmp_path) == "numpy==2.0.0\npyyaml==6.0.1\n"
    assert seen["cwd"] == tmp_path and seen["check"] is True
    assert seen["command"][2:] == ["uv", *requirements.EXPORT]
