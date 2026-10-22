"""Fail when a Python or shell file git would track has more than MAX_LINES lines."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

MAX_LINES = 200


def candidates(root: Path) -> tuple[str, ...]:
    """Tracked files and untracked ones git does not ignore, as posix paths under ``root``."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py", "*.sh"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    # A tracked file deleted from the working tree is still listed.
    return tuple(name for name in sorted(set(listed.split("\0"))) if name and (root / name).is_file())


def over_limit(root: Path) -> dict[str, int]:
    counted = {name: len((root / name).read_bytes().splitlines()) for name in candidates(root)}
    return {name: lines for name, lines in counted.items() if lines > MAX_LINES}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    arguments = parser.parse_args(argv)
    over = over_limit(arguments.root)
    for name, lines in over.items():
        print(f"{name}: {lines} lines (limit {MAX_LINES})")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
