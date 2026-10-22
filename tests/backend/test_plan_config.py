from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from backend import files, schemas
from backend import plan as planner

from .conftest import build

Maker = Callable[..., Any]


@pytest.fixture(autouse=True)
def _models_installed(install_models: Maker) -> None:
    install_models("qwen_image_2_1")


@pytest.mark.parametrize(
    ("model", "trigger", "expected"),
    [
        ("qwen_image_2_1", "  sks person ", ("sks person", "a photo of sks person")),
        ("zimage", "ohwx", ("ohwx", "photo of a woman")),
        ("qwen_image_2_1", "   ", ("alice", "a photo of alice")),
        ("zimage", None, ("sks woman", "photo of a woman")),
    ],
)
def test_a_trigger_word_replaces_the_template_trigger_and_maybe_the_caption(
    make_dataset: Maker, model: str, trigger: str | None, expected: tuple[str, str]
) -> None:
    make_dataset()

    _, config = build(model=model, trigger=trigger)

    assert (config["dataset"]["trigger"], config["dataset"]["default_caption"]) == expected


@pytest.mark.parametrize(
    ("optimizer", "rate", "expected"),
    [(None, None, 1e-6), ("adamw8bit", None, 1e-4), ("adamw8bit", 5e-5, 5e-5)],
)
def test_the_learning_rate_defaults_to_the_optimizers_unless_given(
    make_dataset: Maker, optimizer: str | None, rate: float | None, expected: float
) -> None:
    make_dataset()

    plan, config = build(model="qwen_image_2_1", optimizer=optimizer, learning_rate=rate)

    assert (plan.learning_rate, config["train"]["learning_rate"]) == (expected, expected)
    assert config["train"]["optimizer"] == (optimizer or "automagic3")


@pytest.mark.parametrize(("model", "save_every", "expected"), [("qwen_image_2_1", None, 250), ("zimage", 40, 40)])
def test_checkpoints_are_saved_at_the_templates_interval_unless_given(
    make_dataset: Maker, model: str, save_every: int | None, expected: int
) -> None:
    make_dataset()

    plan, config = build(model=model, save_every=save_every)

    assert (plan.save_every, config["train"]["save_every"]) == (expected, expected)


def test_the_time_estimate_adds_the_setup_time_to_the_measured_step_time(make_dataset: Maker) -> None:
    make_dataset()

    plan, _ = build(model="qwen_image_2_1", max_steps=1000)

    assert plan.estimated_minutes == round((1000 * 0.62 + 20.0) / 60.0, 1) == 10.7


@pytest.mark.parametrize(("model", "dry_run"), [("qwen_image_2_1", True), ("zimage", False)])
def test_there_is_no_time_estimate_for_a_dry_run_or_an_unmeasured_model(
    make_dataset: Maker, model: str, dry_run: bool
) -> None:
    make_dataset()

    plan, _ = build(model=model, dry_run=dry_run)

    assert plan.estimated_minutes is None


def test_missing_checkpoints_are_warned_about_with_the_models_folder_and_returned(
    make_dataset: Maker, workspace: Path
) -> None:
    make_dataset()

    plan, _, _, missing = planner.build_plan(schemas.StartTrainRequest(dataset="alice", model="flux3_klein"))

    assert missing == [
        "diffusion_models/flux3_image_klein_int8_convrot.safetensors",
        "vae/flux2_vae.safetensors",
        "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    ]
    assert plan.warnings[0] == f"Model files not found under {(workspace / 'models').resolve()}: {', '.join(missing)}"


def test_missing_checkpoints_without_a_models_folder_point_at_the_environment_variable(
    make_dataset: Maker, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_dataset()
    monkeypatch.delenv(files.COMFYUI_MODELS_ENV)

    plan, _ = build(model="flux3_klein")

    assert plan.warnings[0].startswith("Model files not found under (set COMFYUI_MODELS_PATH): ")


def test_found_checkpoints_are_written_into_the_model_section(make_dataset: Maker, install_models: Maker) -> None:
    make_dataset()
    paths = install_models("flux3_klein", nested="v1")

    plan, config = build(model="flux3_klein")

    assert [config["model"][key] for key in ("transformer_path", "vae_path", "text_encoder_path")] == [
        str(path) for path in paths
    ]
    assert plan.warnings == []


def test_a_run_that_reuses_a_folder_name_gets_a_numbered_output_folder(make_dataset: Maker, workspace: Path) -> None:
    make_dataset()
    base = workspace / "output" / "qwen_image_2_1"
    (base / "alice-automagic3-500steps").mkdir(parents=True)
    (base / "alice-automagic3-500steps-2").mkdir()

    plan, config = build(model="qwen_image_2_1", max_steps=500)

    assert plan.output_path == str(base / "alice-automagic3-500steps-3")
    assert config["output"] == {**config["output"], "name": "qwen21_alice-automagic3-500steps-3"}


@pytest.mark.parametrize(
    ("body", "folder"),
    [
        ({"max_steps": 500}, "alice-automagic3-500steps"),
        ({"duration_mode": "epochs", "epochs": 30}, "alice-automagic3-30ep"),
    ],
)
def test_a_per_run_model_writes_to_a_folder_named_after_the_run(
    make_dataset: Maker, workspace: Path, body: dict[str, Any], folder: str
) -> None:
    make_dataset()

    plan, config = build(model="qwen_image_2_1", **body)

    expected = workspace / "output" / "qwen_image_2_1" / folder
    assert (plan.output_path, config["output"]["path"], config["output"]["name"]) == (
        str(expected),
        str(expected),
        f"qwen21_{folder}",
    )
    assert config["dataset"]["path"] == str(workspace / "datasets" / "alice")
    assert not expected.exists()


def test_a_legacy_model_writes_straight_to_a_folder_named_after_the_dataset(
    make_dataset: Maker, workspace: Path
) -> None:
    make_dataset()

    plan, config = build(model="zimage")

    assert (plan.output_path, config["output"]["path"]) == (str(workspace / "output" / "alice"),) * 2
