"""Make the top-level modules in scripts/ importable (pyproject's ruff `src` does not affect pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
