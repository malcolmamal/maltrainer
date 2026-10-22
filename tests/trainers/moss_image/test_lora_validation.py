"""What the LoRA trainer refuses instead of quietly doing something else."""

from __future__ import annotations

import pytest

from .runs import prepared_lora, run_lora
from .world import Workspace


@pytest.mark.parametrize("name", ["sgd", "adamww", "lion"])
def test_an_unknown_optimizer_is_refused_not_replaced_by_adamw(workspace: Workspace, name: str) -> None:
    workspace.add_images("p", 2)

    with pytest.raises(ValueError, match=rf"Unknown train\.optimizer '{name}': use automagic3, prodigy, adamw8bit"):
        prepared_lora(workspace.lora_config(train={"optimizer": name}))


def test_adamw_is_still_an_optimizer_choice(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    trainer = prepared_lora(workspace.lora_config(train={"optimizer": "AdamW"}))

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert isinstance(resolved, dict)
    assert resolved["name"] == "adamw"


def test_a_dataset_whose_images_all_fail_to_load_stops_the_run_instead_of_saving_an_untrained_lora(
    workspace: Workspace,
) -> None:
    (workspace.data / "p").mkdir(parents=True)
    (workspace.data / "p" / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"No usable images: every file in the dataset failed to load"):
        run_lora(workspace.lora_config())

    assert list(workspace.out.glob("*.safetensors")) == []
