"""Fail when mypy reports an error anywhere in the checked tree.

The configuration, including the reason for every ``ignore_missing_imports``, is in pyproject.toml.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

# Directories and root-level files; globs are expanded against the root.
TARGETS = (
    "maltrainer",
    "maltrainer_quality",
    "backend",
    "scripts",
    "train_*.py",
    "check_weights.py",
    "compare_loras.py",
    "inspect_lora.py",
)
ERROR = re.compile(r"^(?P<path>.+?):\d+(?::\d+)?: error: ")


def targets(root: Path) -> list[str]:
    found: list[str] = []
    for target in TARGETS:
        found += sorted(path.name for path in root.glob(target)) if "*" in target else [target]
    return found


def mypy_output(root: Path) -> str:
    if importlib.util.find_spec("mypy") is None:
        raise RuntimeError(f"cannot run mypy: not installed for {sys.executable}")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-pretty",
            "--no-error-summary",
            # one platform everywhere, the one CI runs: msvcrt / termios code would check differently per OS
            "--platform",
            "linux",
            "--config-file",
            "pyproject.toml",
            *targets(root),
        ],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
    )
    # 0: clean, 1: type errors. Anything else is mypy failing to run, which must not look like a clean tree.
    if done.returncode not in (0, 1):
        raise RuntimeError(f"mypy exited with {done.returncode}:\n{done.stdout.decode('utf-8', 'replace')}")
    return done.stdout.decode("utf-8", "replace")


def errors_per_file(output: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        matched = ERROR.match(line)
        if matched is not None:
            counts[Path(matched["path"]).as_posix()] += 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    arguments = parser.parse_args(argv)
    errors = errors_per_file(mypy_output(arguments.root))
    for path, count in errors.items():
        print(f"{path}: {count} type errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
