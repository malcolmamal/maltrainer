"""Fail when a function's cyclomatic complexity exceeds MAX_COMPLEXITY.

ruff's C901 measures it. A function is keyed as ``path::name``; a second function of the same
name in the same file is ``path::name#2``, numbered in file order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MAX_COMPLEXITY = 10
MESSAGE = re.compile(r"^`(?P<name>.+)` is too complex \((?P<value>\d+) > \d+\)$")


def findings(root: Path) -> list[dict[str, Any]]:
    listed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--exit-zero",
            "--select",
            "C901",
            "--config",
            f"lint.mccabe.max-complexity = {MAX_COMPLEXITY}",
            "--output-format",
            "json",
            ".",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    found: list[dict[str, Any]] = json.loads(listed)
    return found


def over_limit(root: Path) -> dict[str, int]:
    located = []
    for finding in findings(root):
        matched = MESSAGE.match(finding["message"])
        if matched is None:
            raise ValueError(f"unexpected C901 message: {finding['message']!r}")
        path = Path(os.path.relpath(finding["filename"], root.resolve())).as_posix()
        located.append((path, int(finding["location"]["row"]), matched["name"], int(matched["value"])))
    over: dict[str, int] = {}
    for path, _row, name, value in sorted(located):
        key = f"{path}::{name}"
        number = 1
        while key in over:
            number += 1
            key = f"{path}::{name}#{number}"
        over[key] = value
    return over


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    arguments = parser.parse_args(argv)
    over = over_limit(arguments.root)
    for key, value in over.items():
        print(f"{key}: {value} complexity (limit {MAX_COMPLEXITY})")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
