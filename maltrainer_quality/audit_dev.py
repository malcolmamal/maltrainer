"""Audit the dev tools (the ``dev`` dependency group) for known vulnerabilities.

requirements.lock holds the runtime half only, so the ``audit`` gate never sees ruff, pytest, mypy and the
other tools the gate runs with. This exports the ``dev`` group from uv.lock to a temporary file and hands
it to pip-audit the same way: pinned, without resolving dependencies and without pip.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

EXPORT = ("export", "--frozen", "--only-group", "dev", "--no-hashes", "--no-emit-project")
AUDIT = ("--no-deps", "--disable-pip")


def exported(root: Path) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "uv", *EXPORT],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return done.stdout.decode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path()
    with tempfile.TemporaryDirectory() as folder:
        pinned = Path(folder) / "dev-requirements.txt"
        pinned.write_text(exported(root), encoding="utf-8", newline="\n")
        command = [sys.executable, "-m", "pip_audit", "-r", str(pinned), *AUDIT]
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
