"""Which mutation targets a change touches: the git side of ``mutation --changed``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from maltrainer_quality.mutation_targets import TARGETS, Target


def _git(root: Path, *arguments: str) -> list[str]:
    done = subprocess.run(["git", *arguments], cwd=root, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(arguments)} failed: {done.stderr.strip()}")
    return [line for line in done.stdout.splitlines() if line]


def changed_files(root: Path, base: str) -> set[str]:
    """Paths changed since ``base`` split from this branch: commits, working tree and untracked files."""
    fork = _git(root, "merge-base", "HEAD", base)[0]
    changed = _git(root, "diff", "--name-only", fork)
    return set(changed) | set(_git(root, "ls-files", "--others", "--exclude-standard"))


def select_changed(changed: set[str]) -> list[Target]:
    """Targets whose module or any covering test file is among ``changed``."""
    return [target for target in TARGETS if changed & {target.path, *target.tests}]
