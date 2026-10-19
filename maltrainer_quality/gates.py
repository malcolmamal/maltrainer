"""Run every quality gate and report all of them.

No gate short-circuits another: a run that stops at the first lint error hides the failing test
behind it, so every gate runs and the exit code is decided at the end. Each tool runs as
``python -m <tool>`` under the interpreter running this gate, so the gate checks the environment
it was launched from. A tool that is not installed there fails its gate; it is never skipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Gate:
    name: str
    why: str
    module: str
    arguments: tuple[str, ...] = ()

    @property
    def command(self) -> tuple[str, ...]:
        return (sys.executable, "-m", self.module, *self.arguments)


def gates(junit_dir: Path | None) -> tuple[Gate, ...]:
    pytest: tuple[str, ...] = ("--cov", "--cov-report=term-missing")
    if junit_dir is not None:
        pytest += (f"--junitxml={junit_dir / 'pytest.xml'}", f"--cov-report=xml:{junit_dir / 'coverage.xml'}")
    return (
        Gate("lint", "ruff, with the rule families in pyproject.toml", "ruff", ("check", ".")),
        Gate("format", "formatting is decided by the tool, not by review", "ruff", ("format", "--check", ".")),
        Gate("complexity", "complexity 10 per function, beyond the baseline", "maltrainer_quality.complexity"),
        Gate(
            "file length", "200 lines per Python or shell file, beyond the baseline", "maltrainer_quality.file_length"
        ),
        Gate("tests", "behaviour", "pytest", pytest),
        # pytest-cov can print the shortfall without failing the session; `coverage report`
        # exits non-zero on its own.
        Gate("coverage", "the floor in pyproject.toml", "coverage", ("report",)),
    )


def installed(module: str) -> bool:
    top = module.partition(".")[0]
    return importlib.util.find_spec(top) is not None


def run(gate: Gate) -> bool:
    print(f"\n=== {gate.name}: {gate.why}")
    # The command writes to the inherited descriptor directly. On a pipe, as in CI, anything
    # still buffered here would land below the command's output.
    print(f"$ python {shlex.join(gate.command[1:])}", flush=True)
    if not installed(gate.module):
        print(f"cannot run {gate.module}: not installed for {sys.executable}")
        passed = False
    else:
        passed = subprocess.run(gate.command, check=False).returncode == 0
    print(f"=== {gate.name}: {'ok' if passed else 'FAILED'}")
    return passed


def run_all(table: Iterable[Gate]) -> int:
    failed = [gate.name for gate in table if not run(gate)]
    print()
    if failed:
        print(f"Failed gates: {', '.join(failed)}")
        return 1
    print("All quality gates passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maltrainer_quality", description=__doc__)
    parser.add_argument("--junit", type=Path, help="write pytest and coverage XML here, for CI to publish")
    arguments = parser.parse_args(argv)
    if arguments.junit is not None:
        arguments.junit.mkdir(parents=True, exist_ok=True)
    return run_all(gates(arguments.junit))
