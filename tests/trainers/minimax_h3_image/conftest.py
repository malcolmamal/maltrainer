"""Fixtures for the H3 Image trainer: the H3 trainer's tiny world, with an image checkpoint in place of H3's."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3_image.model import MiniMaxH3ImageTransformer
from maltrainer.trainers.minimax_h3 import model
from tests.maltrainer.models.minimax_h3.h3_tiny import tiny_config
from tests.trainers.minimax_h3.conftest import untouched_process, workspace  # noqa: F401  (fixtures)
from tests.trainers.minimax_h3.fakes import VIDEO_CHANNELS, Fakes, install_fakes, skip_housekeeping

LEVELS = 6


def image_state() -> dict[str, torch.Tensor]:
    """The tiny image checkpoint's tensors (a fresh copy each call)."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        net = MiniMaxH3ImageTransformer(tiny_config(in_channels=VIDEO_CHANNELS), LEVELS)
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    skip_housekeeping(monkeypatch)
    installed = install_fakes(monkeypatch)
    monkeypatch.setattr(model, "load_comfy_int8_convrot_state_dict", lambda _path, **_options: (image_state(), 0))
    return installed
