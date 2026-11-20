"""The side-by-sides: people only, at the higher rate on their own steps, and the optimizers that can't."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from maltrainer.utils.lora_metadata import read_metadata

from .runs import prepared_subjects, run_subjects
from .subjects_world import PLAIN_SGD, sbs, subject, two
from .world import Workspace, write_image


def test_side_by_sides_are_made_for_people_only_and_train_at_the_higher_rate_on_their_own_steps(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolation = two(workspace, first=1, second=1, balance=False, side_by_side=sbs(workspace, lr_boost=3.0))
    isolation["subjects"].append(subject(workspace, "c", "rexo", "rexo, a dog running", 1))
    trainer = prepared_subjects(workspace.lora_config(isolation=isolation, train={**PLAIN_SGD, "max_steps": 1}))
    rates: list[float] = []
    original = trainer.optimizer.step

    def step(*args: Any, **kwargs: Any) -> Any:
        rates.append(trainer.optimizer.param_groups[0]["lr"])
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", step)
    names = [Path(key).name for key, _ in trainer.samples]

    assert sorted(n for n in names if n.startswith("maltrainer_sbs_")) == [
        f"maltrainer_sbs_{i}_0{k}.png" for i in (0, 1) for k in (0, 1)
    ]
    assert len(trainer.composites) == 4 and (workspace.out / "side_by_side").is_dir()
    composite = next(c for _, c in trainer.samples if "stranger" in c)
    assert "zwxemwoman on the" in composite or "qrxtman on the" in composite
    own = next(i for i, (key, _) in enumerate(trainer.samples) if key in trainer.composites)
    other = next(i for i, (key, _) in enumerate(trainer.samples) if key not in trainer.composites)

    trainer._micro_step(own)
    trainer._optimizer_step(trainer.accelerator.unwrap_model(trainer.lora_network))
    trainer._micro_step(other)
    trainer._optimizer_step(trainer.accelerator.unwrap_model(trainer.lora_network))

    assert rates == [pytest.approx(3e-3), pytest.approx(1e-3)]
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_a_warmup_scheduler_keeps_the_rate_it_sets_after_a_boosted_step(workspace: Workspace) -> None:
    isolation = two(workspace, first=1, second=1, side_by_side=sbs(workspace, lr_boost=2.0))
    train = {**PLAIN_SGD, "learning_rate_warmup_steps": 4}
    trainer = prepared_subjects(workspace.lora_config(isolation=isolation, train=train))
    own = next(i for i, (key, _) in enumerate(trainer.samples) if key in trainer.composites)

    trainer._micro_step(own)
    trainer._optimizer_step(trainer.accelerator.unwrap_model(trainer.lora_network))

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 2 / 4)  # warmup step 2 of 4, never boosted


def test_automagic_adapts_its_own_rates_so_the_boost_is_refused_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    isolation = two(workspace, first=1, second=1, side_by_side=sbs(workspace))

    with caplog.at_level(logging.WARNING):
        trainer = prepared_subjects(workspace.lora_config(isolation=isolation))
    own = next(i for i, (key, _) in enumerate(trainer.samples) if key in trainer.composites)
    trainer._micro_step(own)
    before = trainer.optimizer.param_groups[0]["lr"]
    trainer._optimizer_step(trainer.accelerator.unwrap_model(trainer.lora_network))

    assert "lr_boost is ignored with Automagic3" in caplog.text
    assert trainer.optimizer.param_groups[0]["lr"] == before


def test_a_full_run_with_side_by_sides_and_masks_trains_and_records_the_subjects(workspace: Workspace) -> None:
    isolation = two(workspace, side_by_side=sbs(workspace))
    write_image(workspace.root / "a" / "a0-masklabel.png")
    Image.new("L", (32, 32), 255).save(workspace.root / "a" / "a0-masklabel.png")
    config = workspace.lora_config(isolation=isolation, masks={"enabled": True}, train={"max_steps": 3, **PLAIN_SGD})

    trainer = run_subjects(config)

    assert len(trainer.mask_cache) == 1 and torch.isfinite(torch.tensor(trainer.last_dop_loss or 0.0))
    assert json.loads(read_metadata(workspace.out / "k2.safetensors")["maltrainer_isolation_balance"]) == [3, 1]
