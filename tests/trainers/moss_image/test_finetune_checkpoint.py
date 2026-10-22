"""The checkpoint a finetune saves: the layout's keys, the right weights, the right dtype and metadata."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.utils.lora_metadata import read_metadata

from .runs import run_finetune, same_values
from .world import PREFIX, Workspace, named_weights


def _saved(ws: Workspace, name: str = "ft.safetensors") -> dict[str, torch.Tensor]:
    return load_file(str(ws.out / name))


def _module_part(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[len(PREFIX) :]: value for key, value in tensors.items() if not key.endswith(("bands", "config"))}


def test_a_saved_model_has_the_layouts_keys_and_dtypes_and_copies_what_training_does_not_use(
    workspace: Workspace,
) -> None:
    workspace.add_images("p", 2)

    run_finetune(workspace.config())

    saved, layout = _saved(workspace), load_file(str(workspace.layout))
    assert saved.keys() == layout.keys()
    assert {key: value.dtype for key, value in saved.items()} == {key: value.dtype for key, value in layout.items()}
    assert torch.equal(saved["diffusion_model.rope.bands"], layout["diffusion_model.rope.bands"])
    config_key = "diffusion_model.layers.0.comfy_attention.config"
    assert torch.equal(saved[config_key], layout[config_key])


def test_without_ema_the_checkpoint_holds_the_raw_trained_weights(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    trainer = run_finetune(workspace.config())

    assert same_values(_module_part(_saved(workspace)), named_weights(trainer.transformer))


def test_with_ema_the_checkpoint_holds_the_average_where_it_trained_and_the_raw_weights_elsewhere(
    workspace: Workspace,
) -> None:
    workspace.add_images("p", 2)

    trainer = run_finetune(workspace.config(train={"ema_decay": 0.5, "ema_warmup": False}))

    assert trainer.ema is not None
    saved, raw = _module_part(_saved(workspace)), named_weights(trainer.transformer)
    for name, average in trainer.ema.averaged().items():
        assert torch.equal(saved[name], average.to(torch.bfloat16))
        assert not torch.equal(saved[name], raw[name])
    untrained = raw.keys() - trainer.ema.averaged().keys()
    assert untrained and all(torch.equal(saved[name], raw[name]) for name in untrained)


def test_the_output_dtype_sets_the_weights_dtype_but_not_the_copied_entries(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config(output={"dtype": "float32"}))

    saved = _saved(workspace)
    assert {value.dtype for value in _module_part(saved).values()} == {torch.float32}
    assert saved["diffusion_model.layers.0.comfy_attention.config"].dtype == torch.uint8


def test_metadata_records_the_run_the_resolved_optimizer_and_the_config(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    config = workspace.config(train={"ema_decay": 0.5, "ema_every": 2})

    trainer = run_finetune(config)

    meta = read_metadata(workspace.out / "ft.safetensors")
    assert (meta["maltrainer_architecture"], meta["architecture"]) == ("moss_image_finetune", "moss_image_finetune")
    assert meta["maltrainer_git_commit"] == "abc1234" and meta["maltrainer_started_at"]
    assert (meta["maltrainer_trainer"], meta["step"], meta["maltrainer_global_step"]) == (
        "train_moss_image_finetune.py",
        "final",
        "3",
    )
    assert json.loads(meta["maltrainer_optimizer_resolved"]) == trainer.run_metadata["optimizer_resolved"]
    assert json.loads(meta["maltrainer_trainable_prefixes"]) == ["layers."]
    assert (meta["maltrainer_linears_only"], meta["maltrainer_save_dtype"]) == ("true", "bfloat16")
    assert (meta["maltrainer_ema_decay"], meta["maltrainer_ema_every"]) == ("0.5", "2")
    assert (meta["maltrainer_steps_per_epoch"], meta["maltrainer_planned_total_steps"]) == ("2", "3")
    assert json.loads(meta["maltrainer_base_load"])["format"] == "bfloat16"
    assert json.loads(meta["maltrainer_config"]) == json.loads(json.dumps(config))


def test_embed_config_false_leaves_the_config_out_of_the_metadata(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config(output={"embed_config": False}))

    assert "maltrainer_config" not in read_metadata(workspace.out / "ft.safetensors")


@pytest.mark.parametrize("name", ["ft", "ft.safetensors"])
def test_save_every_writes_numbered_models_before_the_end_and_a_final_one(workspace: Workspace, name: str) -> None:
    workspace.add_images("p", 2)

    trainer = run_finetune(workspace.config(output={"name": name}, train={"save_every": 2, "max_steps": 4}))

    saved = sorted(path.name for path in workspace.out.glob("*.safetensors"))
    assert saved == ["ft.safetensors", "ft_step_2.safetensors"]
    assert trainer.last_saved_path == str(workspace.out / "ft.safetensors")
    assert read_metadata(workspace.out / "ft_step_2.safetensors")["step"] == "2"


def test_the_name_defaults_to_moss_image_finetune(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    config = workspace.config()
    del config["output"]["name"]

    run_finetune(config)

    assert (workspace.out / "moss_image_finetune.safetensors").exists()
