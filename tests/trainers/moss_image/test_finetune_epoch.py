"""What one epoch contains: dataset.repeats, dataset.folder_repeats and the step plan built from them."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from .runs import cached_finetune, run_finetune
from .world import Workspace


def _per_folder(trainer: Any) -> dict[str, int]:
    counts = Counter(Path(trainer.samples[i][0]).parent.name for i in trainer._epoch_order())
    return dict(sorted(counts.items()))


def _three_folders(ws: Workspace, **dataset: Any) -> Any:
    ws.add_images("people/ben", 2)
    ws.add_images("people/emma", 4)
    ws.add_images("zoe", 3)
    return cached_finetune(ws.config(dataset=dataset))


def test_repeats_shows_every_image_that_many_times_per_epoch(workspace: Workspace) -> None:
    trainer = _three_folders(workspace, repeats=2)

    assert sorted(Counter(trainer._epoch_order()).values()) == [2] * 9
    assert (trainer._epoch_size(), len(trainer._epoch_order())) == (18, 18)


def test_folder_repeats_match_a_folders_path_or_its_own_name_and_fall_back_to_repeats(workspace: Workspace) -> None:
    trainer = _three_folders(workspace, repeats=1, folder_repeats={"people/ben": 3, "emma": 2})

    assert _per_folder(trainer) == {"ben": 6, "emma": 8, "zoe": 3}
    assert trainer._epoch_size() == 17


def test_a_path_entry_wins_over_a_folder_name_entry(workspace: Workspace) -> None:
    trainer = _three_folders(workspace, folder_repeats={"ben": 5, "people/ben": 2})

    assert _per_folder(trainer)["ben"] == 4


def test_a_fractional_repeat_draws_a_random_part_of_the_folder_each_epoch(workspace: Workspace) -> None:
    trainer = _three_folders(workspace, folder_repeats={"emma": 0.5, "zoe": 2.5})

    first, second = trainer._epoch_order(), trainer._epoch_order()

    assert _per_folder(trainer) == {"ben": 2, "emma": 2, "zoe": 8}
    assert sorted(Counter(i for i in first if "zoe" in trainer.samples[i][0]).values()) == [2, 3, 3]
    assert first != second
    assert trainer._epoch_size() == 12


def test_a_folder_at_zero_repeats_is_cached_but_never_trained(workspace: Workspace) -> None:
    trainer = _three_folders(workspace, folder_repeats={"zoe": 0})

    assert len(trainer.samples) == 9
    assert _per_folder(trainer) == {"ben": 2, "emma": 4}


def test_the_step_plan_counts_an_epoch_with_its_repeats(workspace: Workspace) -> None:
    workspace.add_images("p", 3)

    trainer = run_finetune(workspace.config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2}))

    assert trainer.run_metadata["steps_per_epoch"] == 6
    assert (trainer.run_metadata["planned_total_steps"], trainer.global_step) == (12, 12)


@pytest.mark.parametrize(("steps", "epochs"), [(5, 2), (6, 2), (7, 3)])
def test_the_run_stops_at_max_steps_wherever_it_falls_in_an_epoch(
    workspace: Workspace, steps: int, epochs: int
) -> None:
    workspace.add_images("p", 3)

    trainer = run_finetune(workspace.config(train={"max_steps": steps}))

    assert (trainer.global_step, trainer.epoch + 1) == (steps, epochs)
