"""Every test starts with no GIT_* variable from the process that ran pytest.

`git rebase --exec` in a linked worktree exports GIT_DIR, so a test that runs git in a temporary
folder would otherwise act on the checkout's repository. A test that needs one sets it itself.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def no_inherited_git(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
