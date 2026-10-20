"""Every refusal a finetune config can meet, with its message."""

from __future__ import annotations

from typing import Any

import pytest
from accelerate import Accelerator
from safetensors.torch import load_file, save_file

import train_moss_image_finetune as finetune
from tests.maltrainer.models.moss_image.tiny import int8_tensors, tiny_model

from .runs import cached_finetune, prepared_finetune, run_finetune
from .world import PREFIX, Workspace

PRIOR_WITHOUT_CLASSES = {"weight": 1.0}


def _build(config: dict[str, Any]) -> finetune.MOSSImageFinetuneTrainer:
    return finetune.MOSSImageFinetuneTrainer(config)


@pytest.mark.parametrize(("batch_size", "ok"), [(1, True), (None, True), (0, True), (2, False)])
def test_only_a_batch_size_of_one_is_accepted(workspace: Workspace, batch_size: int | None, ok: bool) -> None:
    config = workspace.config(train={"batch_size": batch_size})

    if ok:
        assert _build(config).config is config
    else:
        with pytest.raises(
            ValueError, match=r"MOSS-Image trains at batch_size 1 .* use train\.gradient_accumulation_steps"
        ):
            _build(config)


def test_finetuning_refuses_more_than_one_process(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Accelerator, "num_processes", property(lambda _self: 2))

    with pytest.raises(ValueError, match=r"MOSS-Image finetuning runs on one GPU"):
        _build(workspace.config())


def test_a_prior_without_classes_or_a_class_word_is_refused(workspace: Workspace) -> None:
    config = workspace.config(prior=PRIOR_WITHOUT_CLASSES)
    del config["prior"]["classes"]

    with pytest.raises(
        ValueError, match=r"prior\.weight > 0 needs prior\.classes .* or dataset\.trigger with prior\.class_word"
    ):
        _build(config)


def test_a_class_word_with_the_dataset_trigger_stands_in_for_classes(workspace: Workspace) -> None:
    config = workspace.config(prior={"weight": 1.0, "class_word": "woman"})
    del config["prior"]["classes"]

    trainer = _build(config)

    assert trainer.prior_classes == {"zed": "woman"}


def test_a_dataset_folder_without_images_is_refused(workspace: Workspace) -> None:
    workspace.data.mkdir()

    with pytest.raises(ValueError, match=r"No images found under .*data"):
        cached_finetune(workspace.config())


def test_a_missing_dataset_folder_is_refused_as_having_no_images(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"No images found under .*data"):
        cached_finetune(workspace.config())


def test_the_int8_checkpoint_is_refused_as_a_base(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    save_file({PREFIX + k: v.contiguous() for k, v in int8_tensors(tiny_model()).items()}, str(workspace.layout))

    with pytest.raises(ValueError, match=r"base\.safetensors is int8; finetuning needs the plain bf16 DiT"):
        prepared_finetune(workspace.config())


def test_a_prefix_that_matches_no_parameter_is_refused(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    with pytest.raises(
        ValueError, match=r"finetune\.trainable_prefixes \['layer\.'\] with linears_only matched no parameter"
    ):
        prepared_finetune(workspace.config(finetune={"trainable_prefixes": ["layer."]}))


def test_linears_only_without_any_linear_under_the_prefix_is_refused(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    with pytest.raises(ValueError, match=r"matched no parameter"):
        prepared_finetune(workspace.config(finetune={"trainable_prefixes": ["final_norm_video."]}))


def test_a_missing_base_checkpoint_stops_the_run_with_the_files_error(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    config = workspace.config(model={"transformer_path": str(workspace.root / "gone.safetensors")})

    with pytest.raises(FileNotFoundError, match=r"gone\.safetensors"):
        run_finetune(config)


def test_a_layout_file_without_an_entry_for_a_model_weight_is_refused_when_saving(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    run_finetune(workspace.config(train={"max_steps": 1}))
    short = {k: v for k, v in load_file(str(workspace.layout)).items() if not k.endswith("video_embedder.bias")}
    save_file(short, str(workspace.layout))
    resume = workspace.config(train={"max_steps": 2, "resume_from": str(workspace.out / "ft_state")})

    with pytest.raises(ValueError, match=r"base\.safetensors has no entry for \['video_embedder\.bias'\]"):
        run_finetune(resume)


def test_the_timestep_distribution_must_be_known(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be logit_normal or uniform, got 'beta'"):
        run_finetune(workspace.config(train={"timestep_distribution": "beta"}))


def test_a_prior_whose_triggers_no_caption_names_is_refused_before_the_first_step(workspace: Workspace) -> None:
    workspace.add_images("p", 1, caption="a photo of someone")
    config = workspace.config(prior={"weight": 1.0, "classes": {"zed": "person"}})

    with pytest.raises(
        ValueError, match=r"prior\.weight > 0, but no caption names a trigger of prior\.classes \['zed'\]"
    ):
        run_finetune(config)

    assert not (workspace.out / "ft.safetensors").exists()
