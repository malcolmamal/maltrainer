"""requirements.lock is a generated file, so a stale one must fail the gate and ``--write`` must fix it."""

from __future__ import annotations

from pathlib import Path

import pytest

from maltrainer_quality import requirements

EXPORT = "# generated\nnumpy==2.0.0\npyyaml==6.0.1\n"


@pytest.fixture(autouse=True)
def fixed_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requirements, "exported", lambda _root: EXPORT)


def test_a_current_file_passes(tmp_path: Path) -> None:
    (tmp_path / requirements.LOCK).write_text(EXPORT, encoding="utf-8")

    assert requirements.main(["--root", str(tmp_path)]) == 0


def test_a_stale_file_fails_and_shows_the_difference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / requirements.LOCK).write_text("# generated\nnumpy==1.0.0\n", encoding="utf-8")

    assert requirements.main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "-numpy==1.0.0" in output
    assert "+numpy==2.0.0" in output
    assert "--write" in output


def test_a_missing_file_fails(tmp_path: Path) -> None:
    assert requirements.main(["--root", str(tmp_path)]) == 1


def test_write_regenerates_the_file(tmp_path: Path) -> None:
    (tmp_path / requirements.LOCK).write_text("old\n", encoding="utf-8")

    assert requirements.main(["--root", str(tmp_path), "--write"]) == 0
    assert (tmp_path / requirements.LOCK).read_text(encoding="utf-8") == EXPORT
    assert requirements.main(["--root", str(tmp_path)]) == 0


def test_crlf_in_the_checked_out_file_does_not_make_it_stale(tmp_path: Path) -> None:
    (tmp_path / requirements.LOCK).write_bytes(EXPORT.replace("\n", "\r\n").encode("utf-8"))

    assert requirements.main(["--root", str(tmp_path)]) == 0
