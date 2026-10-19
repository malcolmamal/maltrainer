from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from maltrainer_quality import gates
from maltrainer_quality.gates import Gate


def test_every_gate_runs_after_one_fails(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    table = (
        Gate("missing", "a tool that is not installed", "no_such_tool_installed"),
        Gate("passing", "a tool that succeeds", "json.tool", ("--help",)),
        Gate("failing", "a tool that fails", "json.tool", (str(tmp_path / "absent.json"),)),
    )

    assert gates.run_all(table) == 1
    output = capfd.readouterr().out
    assert "cannot run no_such_tool_installed: not installed for" in output
    assert "=== passing: ok" in output
    assert "=== failing: FAILED" in output
    assert output.endswith("\nFailed gates: missing, failing\n")


def test_all_gates_passing_says_so(capfd: pytest.CaptureFixture[str]) -> None:
    assert gates.run_all([Gate("passing", "a tool that succeeds", "json.tool", ("--help",))]) == 0
    assert capfd.readouterr().out.endswith("\nAll quality gates passed.\n")


def test_a_gate_runs_its_tool_under_this_interpreter() -> None:
    assert Gate("lint", "why", "ruff", ("check", ".")).command == (sys.executable, "-m", "ruff", "check", ".")


def test_a_submodule_counts_as_installed_when_its_package_is() -> None:
    assert gates.installed("maltrainer_quality.complexity")
    assert not gates.installed("no_such_tool_installed.module")


def test_the_table_runs_every_gate_in_order() -> None:
    names = [gate.name for gate in gates.gates(None)]

    assert names == ["lint", "format", "complexity", "file length", "tests", "coverage"]


def test_junit_output_goes_to_the_named_folder(tmp_path: Path) -> None:
    tests = next(gate for gate in gates.gates(tmp_path) if gate.name == "tests")

    assert f"--junitxml={tmp_path / 'pytest.xml'}" in tests.arguments
    assert f"--cov-report=xml:{tmp_path / 'coverage.xml'}" in tests.arguments


def test_main_creates_the_junit_folder_and_runs_the_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ran: list[tuple[Gate, ...]] = []
    monkeypatch.setattr(gates, "run_all", lambda table: ran.append(tuple(table)) or 0)

    assert gates.main(["--junit", str(tmp_path / "reports")]) == 0
    assert (tmp_path / "reports").is_dir()
    assert ran == [gates.gates(tmp_path / "reports")]


def test_main_without_junit_runs_the_plain_table(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[tuple[Gate, ...]] = []
    monkeypatch.setattr(gates, "run_all", lambda table: ran.append(tuple(table)) or 0)

    assert gates.main([]) == 0
    assert ran == [gates.gates(None)]


def test_the_package_runs_as_a_module(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["maltrainer_quality", "--help"])

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("maltrainer_quality", run_name="__main__")

    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("usage: maltrainer_quality")
