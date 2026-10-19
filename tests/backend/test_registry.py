from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend import app as backend

from .conftest import REPO_ROOT

SPECS = list(backend.MODELS.values())


def _template_files(spec: backend.ModelSpec) -> list[str]:
    return [spec.template, *(path for _, path in spec.optimizer_templates)]


def test_the_registry_is_keyed_by_each_models_id() -> None:
    assert [(key, spec.id) for key, spec in backend.MODELS.items()] == [(spec.id, spec.id) for spec in SPECS]


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.id)
def test_each_models_training_script_exists_in_the_repo(spec: backend.ModelSpec) -> None:
    assert (REPO_ROOT / spec.script).is_file()


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.id)
def test_each_models_templates_exist_in_the_repo(spec: backend.ModelSpec) -> None:
    assert all((REPO_ROOT / path).is_file() for path in _template_files(spec))


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.id)
def test_each_model_defaults_to_an_optimizer_it_offers(spec: backend.ModelSpec) -> None:
    offered = [optimizer.name for optimizer in spec.optimizers]

    assert spec.default_optimizer in offered
    assert {name for name, _ in spec.optimizer_templates} <= set(offered)


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.id)
def test_every_template_loads_with_the_sections_the_plan_edits(spec: backend.ModelSpec) -> None:
    for optimizer in [None, *(name for name, _ in spec.optimizer_templates)]:
        config = backend.load_template(spec, "alice", optimizer)

        assert {"dataset", "train", "output"} <= set(config)
        assert "[person_to_train]" not in str(config)


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.id)
def test_every_template_names_the_optimizer_it_is_for(spec: backend.ModelSpec) -> None:
    for name, _ in spec.optimizer_templates:
        assert backend.load_template(spec, "alice", name)["train"]["optimizer"] == name
    assert backend.load_template(spec, "alice")["train"]["optimizer"] == spec.default_optimizer


def test_a_template_gets_the_dataset_name_and_the_models_folder(workspace: Path) -> None:
    config = backend.load_template(backend.MODELS["qwen_image_2_1"], "alice")

    assert config["dataset"]["trigger"] == "alice"
    assert config["output"]["name"] == "qwen21_alice"
    assert config["model"]["vae_path"] == f"{(workspace / 'models').as_posix()}/vae/qwen_image_2.1_vae_bf16.safetensors"


def test_a_template_keeps_the_models_placeholder_without_a_models_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(backend.COMFYUI_MODELS_ENV)

    config = backend.load_template(backend.MODELS["qwen_image_2_1"], "alice")

    assert config["model"]["vae_path"].startswith("[comfyui_models]/vae/")


def test_the_optimizers_own_template_replaces_the_default_one() -> None:
    spec = backend.MODELS["krea2"]

    default = backend.load_template(spec, "alice", "automagic3")
    prodigy = backend.load_template(spec, "alice", "prodigy")

    assert (default["lora"]["rank"], prodigy["lora"]["rank"]) == (16, 16)
    assert (default["lora"]["alpha"], prodigy["lora"]["alpha"]) == (16, 1)


def test_a_missing_template_file_is_a_server_error(tweak_spec: Callable[..., backend.ModelSpec]) -> None:
    spec = tweak_spec("zimage", template="templates/nope.yaml")

    with pytest.raises(HTTPException) as error:
        backend.load_template(spec, "alice")

    assert (error.value.status_code, error.value.detail) == (500, "Template not found: templates/nope.yaml")
