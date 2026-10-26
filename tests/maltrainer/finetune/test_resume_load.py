"""A resume file is data: loading it never runs or builds code from the file."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from maltrainer.finetune.resume import load_resume_file


class Arbitrary:
    """Stands for any object a tampered resume file could carry."""


def test_the_plain_data_a_resume_state_holds_round_trips(tmp_path: Path) -> None:
    state = {
        "global_step": 12,
        "epoch_rng": random.Random(3).getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "group_names": ["attn"],
        "optimizers": {
            0: {"state": {0: {"lr": torch.full((2,), 1e-5)}}, "param_groups": [{"lr": 1e-5, "params": [0]}]}
        },
    }
    path = tmp_path / "trainer_state.pt"
    torch.save(state, path)

    loaded = load_resume_file(path, "cpu")

    assert loaded["global_step"] == 12 and loaded["epoch_rng"] == state["epoch_rng"]
    assert torch.equal(loaded["torch_rng"], state["torch_rng"])
    assert torch.equal(loaded["optimizers"][0]["state"][0]["lr"], torch.full((2,), 1e-5))


def test_a_file_carrying_an_object_is_refused_instead_of_unpickled(tmp_path: Path) -> None:
    path = tmp_path / "trainer_state.pt"
    torch.save({"global_step": 1, "extra": Arbitrary()}, path)

    with pytest.raises(ValueError, match=r"trainer_state\.pt.*holds more than plain tensors and numbers"):
        load_resume_file(path, "cpu")
