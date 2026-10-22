#!/usr/bin/env bash
# maltrainer quality gate (Linux / macOS / Git Bash): every gate, as CI runs it.
#
# Uses the dev environment .venv when it exists, then venv, then python3 / python on PATH.
# That interpreter needs the tools in pyproject.toml's dev group. Extra arguments pass through, e.g.
#   ./quality.sh --junit reports

set -u
cd "$(dirname "$0")"

# The first one that runs: on Windows, python3 on PATH can be the Microsoft Store stub.
PYTHON=python
for candidate in .venv/bin/python .venv/Scripts/python.exe venv/bin/python venv/Scripts/python.exe python3 python; do
    if "$candidate" -c "" >/dev/null 2>&1; then
        PYTHON=$candidate
        break
    fi
done

exec "$PYTHON" -m maltrainer_quality "$@"
