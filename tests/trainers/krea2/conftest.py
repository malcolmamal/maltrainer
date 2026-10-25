"""Fixtures: a workspace per test, the fakes, the tiny model config, and a process left as the test found it."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState, GradientState

import train_krea2_lora
from maltrainer.finetune import rotation
from maltrainer.models.krea2 import loading
from maltrainer.trainers.krea2 import finetune_save, previews

from .fakes import Fakes, install_fakes, skip_housekeeping
from .tiny import CONFIG
from .world import Workspace, write_checkpoint


@pytest.fixture(autouse=True)
def untouched_process() -> Iterator[None]:
    """The trainers add log handlers, seed every RNG and start an accelerate singleton; undo all three."""
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
    ws = Workspace(tmp_path)
    write_checkpoint(ws.checkpoint)
    return ws


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """The tiny DiT instead of the 12.9B one the loader builds, instant git and gc, fake VAE and text encoders."""
    monkeypatch.setattr(loading, "Krea2Config", lambda: CONFIG)
    skip_housekeeping(monkeypatch)
    monkeypatch.setattr(rotation, "gc", SimpleNamespace(collect=lambda: 0))  # a window swap collects garbage
    return install_fakes(monkeypatch, train_krea2_lora, previews)


@pytest.fixture(autouse=True)
def disk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The free bytes ``shutil.disk_usage`` reports to the finetune's saves (plenty); set ``.free`` to change it."""
    state = SimpleNamespace(free=1 << 40)
    monkeypatch.setattr(finetune_save.shutil, "disk_usage", lambda _path: SimpleNamespace(free=state.free))
    return state
