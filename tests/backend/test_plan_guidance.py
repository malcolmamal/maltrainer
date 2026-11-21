from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from .conftest import build, write_images

Maker = Callable[..., Any]


@pytest.fixture(autouse=True)
def _models_installed(install_models: Maker) -> None:
    install_models("qwen_image_2_1")


def test_a_run_under_the_benchmarked_epoch_range_is_flagged_as_under_trained(make_dataset: Maker) -> None:
    make_dataset(count=20)

    plan, _ = build(model="qwen_image_2_1", max_steps=300)

    assert plan.warnings == [
        "Only 15.0 epochs. In the benchmark, likeness was at roughly 70% of its plateau after 11 epochs and "
        "90–95% after 22, so runs under ~20 epochs risk being under-trained. Raise max steps if you can."
    ]


@pytest.mark.parametrize("steps", [400, 2400])
def test_the_edges_of_the_benchmarked_epoch_range_are_not_warned_about(make_dataset: Maker, steps: int) -> None:
    make_dataset(count=20)

    plan, _ = build(model="qwen_image_2_1", max_steps=steps)

    assert plan.warnings == []


def test_a_run_past_the_benchmarked_epoch_range_and_step_count_gets_both_warnings(make_dataset: Maker) -> None:
    make_dataset(count=20)

    plan, _ = build(model="qwen_image_2_1", max_steps=2600)

    assert plan.warnings == [
        "130.0 epochs is beyond what was benchmarked (up to 119); very small datasets can over-fit or memorize.",
        "2600 steps is longer than any benchmarked run (2500 at most). Automagic3 stayed stable up to there; "
        "beyond it is untested.",
    ]


def test_a_model_without_benchmark_guidance_is_never_warned_about_length(
    make_dataset: Maker, install_models: Maker
) -> None:
    make_dataset(count=20)
    install_models("krea2")

    plan, _ = build(model="krea2", max_steps=50)

    assert plan.warnings == []


def test_prodigy_on_qwen_warns_that_it_overcooks(make_dataset: Maker) -> None:
    make_dataset(count=20)

    plan, config = build(model="qwen_image_2_1", optimizer="prodigy", max_steps=1500)

    assert plan.warnings == [
        "Prodigy overcooks Qwen-Image 2.1 LoRAs in benchmarks; keep checkpoints at or below ~1500 steps."
    ]
    assert (plan.learning_rate, config["train"]["optimizer"], config["train"]["learning_rate"]) == (1.0, "prodigy", 1.0)


def test_non_square_images_are_counted_and_exemplified_for_square_only_models(make_dataset: Maker) -> None:
    folder = make_dataset(count=1)
    write_images(folder / "wide", 3, (16, 8))

    plan, _ = build(model="qwen_image_2_1", max_steps=1500)

    assert plan.image_count == 4
    assert plan.warnings[0].startswith("3 of 4 images are not square (e.g. 16×8). Training is square 512×512 only")


def test_square_images_and_other_models_get_no_squashing_warning(make_dataset: Maker) -> None:
    folder = make_dataset(count=20)
    write_images(folder / "wide", 2, (16, 8))

    plan, _ = build(model="zimage")

    assert plan.warnings == []


def test_visage_crops_non_square_images_so_the_plan_does_not_say_they_are_squashed(
    make_dataset: Maker, install_models: Maker
) -> None:
    install_models("visage")
    folder = make_dataset(count=1)
    write_images(folder / "wide", 3, (16, 8))

    plan, config = build(model="visage")

    assert plan.image_count == 4 and plan.warnings == []
    assert config["dataset"]["aspect_mode"] == "square"
