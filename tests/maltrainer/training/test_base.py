"""The shared halves of a LoRA trainer: what they leave to the trainer, and the dry run's lines."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.training.loop import StepLoop
from maltrainer.training.progress import simulate_steps


@pytest.mark.parametrize(
    ("cls", "call"),
    [
        (ImageLoRATrainer, lambda t: t.precache_dataset()),
        (ImageLoRATrainer, lambda t: t.load_transformer_and_lora()),
        (StepLoop, lambda t: t._epoch_order()),
        (StepLoop, lambda t: t._training_step(0)),
        (StepLoop, lambda t: t.save_checkpoint(1)),
    ],
)
def test_the_model_side_is_the_trainer_s_to_write(cls: type, call: Callable[[Any], object]) -> None:
    bare = object.__new__(cls)  # the parts in question need no set-up

    with pytest.raises(NotImplementedError):
        call(bare)


def test_a_dry_run_logs_every_step_and_where_it_would_save(caplog: pytest.LogCaptureFixture) -> None:
    progress = logging.getLogger("test_dry_run_progress")

    with caplog.at_level(logging.INFO):
        simulate_steps(progress, (2, 3, 5), 0.0, show=False, save_every=2)

    steps = [r.getMessage() for r in caplog.records if r.name == progress.name]
    assert steps[0] == "Step: 1/5 [20%], Loss: 0.000000, Epoch: 1/3, LR: 0.00e+00 (dry run)"
    assert steps[-1] == "Step: 5/5 [100%], Loss: 0.000000, Epoch: 3/3, LR: 0.00e+00 (dry run)"
    saves = [r.getMessage() for r in caplog.records if r.name == "maltrainer.training.progress"]
    assert saves == [
        "Dry run: would save checkpoint at step 2",
        "Dry run: would save checkpoint at step 4",
        "Dry run complete: would save final checkpoint at step 5",
    ]
