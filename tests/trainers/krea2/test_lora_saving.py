"""save_checkpoint: file names, the output folder, embedded metadata and who writes."""

from __future__ import annotations

from typing import Any

import pytest
from accelerate import Accelerator
from safetensors import safe_open

from .runs import prepared_lora, progress_lines, run_lora, trained_lora
from .world import Workspace


def _files(ws: Workspace) -> list[str]:
    return sorted(path.name for path in ws.out.glob("*.safetensors"))


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"name": "mine"}, "mine.safetensors"),
        ({"name": "mine.safetensors"}, "mine.safetensors"),
        ({"name": None, "lora_name": "legacy"}, "legacy.safetensors"),
        ({"name": None}, "krea2_lora.safetensors"),
    ],
)
def test_the_final_file_is_named_after_output_name_then_lora_name_then_the_default(
    workspace: Workspace, output: dict[str, Any], expected: str
) -> None:
    workspace.add_images(1)

    trainer = run_lora(workspace.lora_config(output=output, train={"max_steps": 1}))

    assert _files(workspace) == [expected]
    assert trainer.last_saved_path == str(workspace.out / expected)


def test_a_numbered_save_appends_the_step_to_the_name(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_lora(workspace.lora_config(output={"name": "mine.safetensors"}))
    trainer.lora_network = trainer.accelerator.prepare(trainer.lora_network)

    trainer.save_checkpoint(7)

    assert _files(workspace) == ["mine_step_7.safetensors"]


def test_the_output_folder_is_created_when_it_is_missing(workspace: Workspace) -> None:
    nested = workspace.root / "a" / "b"

    workspace.add_images(1)

    run_lora(workspace.lora_config(output={"path": str(nested)}, train={"max_steps": 1}))

    assert (nested / "k2.safetensors").exists()


def test_a_process_that_is_not_the_main_one_writes_nothing(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_lora(workspace.lora_config())
    trainer.is_main_process = False

    trainer.save_checkpoint("final")

    assert _files(workspace) == []
    assert trainer.last_saved_path is None


def test_the_metadata_names_the_run_the_model_and_the_save(workspace: Workspace) -> None:
    trained_lora(workspace, train={"max_steps": 1})

    with safe_open(str(workspace.out / "k2.safetensors"), framework="pt") as handle:
        meta = handle.metadata()
    assert meta["maltrainer_git_commit"] == "abc1234" and meta["maltrainer_model_type"] == "krea2"
    assert meta["maltrainer_save_dtype"] == "bfloat16" and meta["maltrainer_dataset_samples"] == "2"
    assert '"fp8_linears": 16' in meta["maltrainer_base_load"]
    assert meta["maltrainer_planned_total_steps"] == "1" and "maltrainer_saved_at" in meta


def test_embed_config_false_leaves_the_config_out_of_the_file(workspace: Workspace) -> None:
    trained_lora(workspace, output={"embed_config": False}, train={"max_steps": 1})

    with safe_open(str(workspace.out / "k2.safetensors"), framework="pt") as handle:
        assert "maltrainer_config" not in handle.metadata()


def test_a_whole_run_on_a_non_main_process_trains_quietly_and_saves_nothing(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Accelerator, "is_main_process", property(lambda _self: False))
    workspace.add_images(1)

    trainer = run_lora(workspace.lora_config(train={"max_steps": 2, "save_every": 1}))

    assert _files(workspace) == [] and trainer.last_saved_path is None
    assert len(progress_lines(workspace)) == 2
