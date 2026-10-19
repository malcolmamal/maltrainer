from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from .conftest import build

Maker = Callable[..., Any]


@pytest.fixture(autouse=True)
def _models_installed(install_models: Maker) -> None:
    for model_id in ("qwen_image_2_1", "krea2"):
        install_models(model_id)


def test_a_whole_epoch_model_defaults_to_a_hundred_epochs(make_dataset: Maker) -> None:
    make_dataset(count=4)

    plan, config = build(model="zimage")

    assert (plan.duration_mode, plan.epochs, plan.steps_per_epoch, plan.total_steps) == ("epochs", 100, 4, 400)
    assert plan.duration == "100 epochs (400 steps)"
    assert (config["train"]["num_epochs"], "max_steps" in config["train"]) == (100, False)
    assert plan.warnings == []


def test_epochs_mode_trains_the_requested_epochs(make_dataset: Maker) -> None:
    make_dataset(count=4)

    plan, _ = build(model="zimage", epochs=30)

    assert (plan.epochs, plan.total_steps) == (30, 120)


def test_a_step_budget_on_a_whole_epoch_model_is_rounded_up_with_a_warning(make_dataset: Maker) -> None:
    make_dataset(count=4)

    plan, config = build(model="zimage", duration_mode="steps", max_steps=450)

    assert (plan.epochs, plan.total_steps, plan.duration) == (113, 452, "113 epochs (452 steps)")
    assert plan.warnings == ["Z-Image (Base) trains whole epochs: 450 steps rounded up to 113 epochs = 452 steps."]
    assert (config["train"]["num_epochs"], "max_steps" in config["train"]) == (113, False)


def test_a_step_budget_that_fits_whole_epochs_is_not_warned_about(make_dataset: Maker) -> None:
    make_dataset(count=4)

    plan, _ = build(model="zimage", duration_mode="steps", max_steps=400)

    assert (plan.epochs, plan.total_steps, plan.warnings) == (100, 400, [])


def test_steps_mode_without_a_budget_uses_the_models_default_and_no_auto_choice(make_dataset: Maker) -> None:
    make_dataset(count=4)

    plan, _ = build(model="zimage", duration_mode="steps")

    assert (plan.total_steps, plan.auto_steps, plan.notes) == (2000, False, [])


def test_a_native_model_gets_its_step_budget_in_the_config(make_dataset: Maker) -> None:
    make_dataset(count=10)

    plan, config = build(model="qwen_image_2_1", max_steps=600)

    assert (plan.total_steps, plan.epochs, plan.duration, plan.auto_steps) == (600, 60, "600 steps", False)
    assert (config["train"]["max_steps"], config["train"]["num_epochs"]) == (600, 60)
    assert plan.notes == []


def test_epochs_mode_on_a_native_model_turns_the_step_budget_off(make_dataset: Maker) -> None:
    make_dataset(count=10)

    plan, config = build(model="qwen_image_2_1", duration_mode="epochs", epochs=30)

    assert (plan.total_steps, plan.duration) == (300, "30 epochs (300 steps)")
    assert (config["train"]["max_steps"], config["train"]["num_epochs"]) == (0, 30)


KEPT = " steps, then kept within 1000–2500 steps, the range the benchmark covered, giving "


@pytest.mark.parametrize(
    ("images", "steps", "note"),
    [
        (10, 1000, f"10 images × ~70 epochs = 700{KEPT}1000."),
        (21, 1450, "21 images × ~70 epochs = 1470 steps, rounded to 1450."),
        (
            12,
            1000,
            "12 images × ~70 epochs = 840 steps, rounded to 850, then kept within 1000–2500 steps, the range the benchmark covered, giving 1000.",
        ),
        (40, 2500, f"40 images × ~70 epochs = 2800{KEPT}2500."),
        (30, 2100, "30 images × ~70 epochs = 2100 steps."),
    ],
)
def test_the_step_budget_is_chosen_from_the_dataset_size_when_none_is_given(
    make_dataset: Maker, images: int, steps: int, note: str
) -> None:
    make_dataset(count=images)

    plan, config = build(model="qwen_image_2_1")

    assert (plan.auto_steps, plan.total_steps, config["train"]["max_steps"]) == (True, steps, steps)
    assert plan.notes == [f"Steps chosen automatically: {note}"]


def test_an_empty_dataset_gets_the_default_budget_and_a_warning(workspace: Path) -> None:
    (workspace / "datasets" / "alice").mkdir()

    plan, _ = build(model="qwen_image_2_1")

    assert (plan.image_count, plan.steps_per_epoch, plan.total_steps, plan.auto_steps) == (0, 1, 1500, False)
    assert plan.warnings == ["The dataset has no images."]


def test_repeats_batch_size_and_gradient_accumulation_set_the_steps_per_epoch(
    make_dataset: Maker, tweak_spec: Maker
) -> None:
    make_dataset(count=10)
    tweak_spec(
        "zimage", sections={"dataset": {"repeats": 3}, "train": {"batch_size": 4, "gradient_accumulation_steps": 2}}
    )

    plan, _ = build(model="zimage", epochs=5)

    assert (plan.steps_per_epoch, plan.total_steps) == (4, 20)


@pytest.mark.parametrize("value", [0, None])
def test_a_zero_or_blank_template_value_counts_as_one(make_dataset: Maker, tweak_spec: Maker, value: Any) -> None:
    make_dataset(count=10)
    tweak_spec(
        "zimage",
        sections={"dataset": {"repeats": value}, "train": {"batch_size": value, "gradient_accumulation_steps": value}},
    )

    plan, _ = build(model="zimage")

    assert plan.steps_per_epoch == 10


def test_the_real_krea2_prodigy_template_accumulates_two_batches_per_step(make_dataset: Maker) -> None:
    make_dataset(count=10)

    plan, config = build(model="krea2", optimizer="prodigy", epochs=4, duration_mode="epochs")

    assert (plan.steps_per_epoch, plan.total_steps) == (5, 20)
    assert (config["lora"]["rank"], config["train"]["gradient_accumulation_steps"]) == (16, 2)
