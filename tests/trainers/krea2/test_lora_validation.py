"""What the LoRA trainer refuses instead of quietly doing something else."""

from __future__ import annotations

import pytest

from .runs import run_lora
from .world import Workspace


def test_a_dataset_whose_images_all_fail_to_load_stops_the_run_instead_of_saving_an_untrained_lora(
    workspace: Workspace,
) -> None:
    workspace.data.mkdir(parents=True)
    (workspace.data / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"No usable images: every file in the dataset failed to load"):
        run_lora(workspace.lora_config())

    assert list(workspace.out.glob("*.safetensors")) == []
