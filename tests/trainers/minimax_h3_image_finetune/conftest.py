"""Fixtures: a workspace with the tiny INT8 checkpoint, the fakes (the real INT8 loader kept), a clean process."""

from __future__ import annotations

from pathlib import Path

import pytest

from maltrainer.models.minimax_h3.int8_convrot import load_comfy_int8_convrot_state_dict
from maltrainer.trainers.minimax_h3 import model
from tests.trainers.minimax_h3.conftest import untouched_process  # noqa: F401  (fixture)
from tests.trainers.minimax_h3.fakes import Fakes, install_fakes, skip_housekeeping
from tests.trainers.minimax_h3.world import Workspace

from .world import write_int8_checkpoint


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path)
    write_int8_checkpoint(ws.root / "transformer.safetensors")
    return ws


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    skip_housekeeping(monkeypatch)
    installed = install_fakes(monkeypatch)
    monkeypatch.setattr(model, "load_comfy_int8_convrot_state_dict", load_comfy_int8_convrot_state_dict)
    return installed
