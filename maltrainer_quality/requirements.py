"""Fail when requirements.lock is not what uv.lock exports.

requirements.lock is uv.lock's runtime half for pip users: every pinned dependency except torch
and what only torch needs (the build for the GPU is installed first, see README). It is a
generated file; ``--write`` regenerates it after ``uv lock``.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

LOCK = Path("requirements.lock")
EXPORT = ("export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project", "--prune", "torch")


def exported(root: Path) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "uv", *EXPORT],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return done.stdout.decode("utf-8").replace("\r\n", "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--write", action="store_true", help="regenerate requirements.lock from uv.lock")
    arguments = parser.parse_args(argv)
    path = arguments.root / LOCK
    wanted = exported(arguments.root)
    if arguments.write:
        path.write_text(wanted, encoding="utf-8", newline="\n")
        return 0
    current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.is_file() else ""
    if current == wanted:
        return 0
    diff = difflib.unified_diff(
        current.splitlines(), wanted.splitlines(), "requirements.lock", "uv export", lineterm=""
    )
    print("\n".join(diff))
    print("requirements.lock is stale. Regenerate it with `python -m maltrainer_quality.requirements --write`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
