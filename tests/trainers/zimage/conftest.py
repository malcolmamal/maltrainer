"""Fixtures: a workspace per test, the fakes, and a process left as the test found it."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState, GradientState

import train_zimage_embedding
import train_zimage_lora

from .fakes import Fakes, install_fakes
from .world import Workspace


@pytest.fixture(autouse=True)
def untouched_process() -> Iterator[None]:
    """The LoRA trainer adds a log handler, both seed every RNG and start an accelerate singleton; undo all."""
    root, progress = logging.getLogger(), logging.getLogger("progress")
    handlers, level, propagate = list(root.handlers), progress.level, progress.propagate
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    for handler in [h for h in root.handlers if h not in handlers] + list(progress.handlers):
        handler.close()
        root.removeHandler(handler)
        progress.removeHandler(handler)
    progress.setLevel(level)
    progress.propagate = propagate
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    return install_fakes(monkeypatch, train_zimage_lora, train_zimage_embedding)
