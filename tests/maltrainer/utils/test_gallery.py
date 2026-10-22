from __future__ import annotations

import contextlib
import random
from pathlib import Path

import pytest
from PIL import Image

from maltrainer.utils.gallery import DEFAULT_PROMPTS, Gallery

PEOPLE = {"anna": "woman", "bob": "man", "cy": "robot", "dee": "woman"}


def _renderer(calls: list[tuple], size: tuple = (8, 6)):
    def render(prompt: str, seed: int, label: str) -> Image.Image:
        calls.append((prompt, seed, label))
        return Image.new("RGB", size, (len(calls), 0, 0))

    return render


@pytest.mark.parametrize(
    ("config", "enabled"),
    [(None, True), (True, True), (False, False), ({}, True), ({"enabled": False}, False)],
)
def test_the_gallery_is_on_unless_switched_off(config: object, enabled: bool, tmp_path: Path) -> None:
    assert Gallery(config, PEOPLE, tmp_path).enabled is enabled


def test_a_finetune_without_named_people_has_no_gallery(tmp_path: Path) -> None:
    gallery = Gallery(None, {}, tmp_path)

    assert not gallery.enabled
    assert gallery.captions == []


@pytest.mark.parametrize(("configured", "count"), [(3, 3), (0, 8), (None, 8), (-4, 1)])
def test_the_count_is_at_least_one_and_defaults_to_eight(configured: object, count: int, tmp_path: Path) -> None:
    assert Gallery({"count": configured}, PEOPLE, tmp_path).count == count


def test_prompts_fill_in_the_trigger_and_pick_templates_by_class_word(tmp_path: Path) -> None:
    gallery = Gallery({"prompts": {"man": ["{trigger} the {class}"]}}, PEOPLE, tmp_path)

    assert gallery.prompts_for("bob") == ["bob the man"]
    assert gallery.prompts_for("anna") == [t.replace("{trigger}", "anna") for t in DEFAULT_PROMPTS["woman"]]
    assert gallery.prompts_for("cy")[0].startswith("photo of cy, a robot smiling")


def test_the_captions_cover_every_person_and_prompt_and_nothing_when_off(tmp_path: Path) -> None:
    gallery = Gallery(None, PEOPLE, tmp_path)

    assert len(gallery.captions) == 4 + 4 + 2 + 4
    assert Gallery(False, PEOPLE, tmp_path).captions == []


def test_a_press_renders_count_distinct_people_and_writes_files(tmp_path: Path) -> None:
    gallery = Gallery({"count": 3}, PEOPLE, tmp_path)
    calls: list[tuple] = []

    gallery.run(120, _renderer(calls), contextlib.nullcontext, label="EMA")

    assert len(calls) == 3
    assert [c[2] for c in calls] == ["gallery 1/3", "gallery 2/3", "gallery 3/3"]
    folder = tmp_path / "samples" / "gallery"
    singles = sorted(p.name for p in folder.glob("step000120_*.png"))
    assert len(singles) == 3
    notes = (folder / "step000120.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(line.split()[0] for line in notes) == [p.split("_", 1)[1][:-4] for p in singles]
    for line, (prompt, seed, _) in zip(notes, calls, strict=False):
        assert line.endswith(f"seed {seed}  {prompt}")
    with Image.open(folder / "step000120.png") as grid:
        assert grid.size == (3 * 8, 6)


def test_the_grid_wraps_after_four_columns(tmp_path: Path) -> None:
    gallery = Gallery({"count": 6}, {f"p{i}": "woman" for i in range(6)}, tmp_path)

    gallery.run(1, _renderer([], size=(5, 4)), contextlib.nullcontext)

    with Image.open(tmp_path / "samples" / "gallery" / "step000001.png") as grid:
        assert grid.size == (4 * 5, 2 * 4)


def test_the_weights_are_in_place_only_while_rendering(tmp_path: Path) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def weights():
        events.append("on")
        yield
        events.append("off")

    def render(_prompt: str, _seed: int, _label: str) -> Image.Image:
        events.append("render")
        return Image.new("RGB", (2, 2))

    Gallery({"count": 2}, PEOPLE, tmp_path).run(5, render, weights)

    assert events == ["on", "render", "render", "off"]


def test_people_whose_prompts_are_not_encoded_are_skipped(tmp_path: Path) -> None:
    gallery = Gallery({"count": 4}, PEOPLE, tmp_path)
    calls: list[tuple] = []

    gallery.run(2, _renderer(calls), contextlib.nullcontext, ready=lambda prompt: "anna" in prompt)

    assert len(calls) == 1
    assert "anna" in calls[0][0]


def test_with_nothing_encoded_a_press_renders_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    calls: list[tuple] = []

    seconds = Gallery(None, PEOPLE, tmp_path).run(2, _renderer(calls), contextlib.nullcontext, ready=lambda _p: False)

    assert (seconds, calls) == (0.0, [])
    assert not (tmp_path / "samples").exists()
    assert "nothing to render" in caplog.text


def test_a_press_leaves_the_global_random_state_alone(tmp_path: Path) -> None:
    random.seed(1234)
    expected = random.random()
    random.seed(1234)

    Gallery(None, PEOPLE, tmp_path).run(3, _renderer([]), contextlib.nullcontext)

    assert random.random() == expected


def test_the_gallery_of_a_config_shows_the_prior_classes(tmp_path: Path) -> None:
    config = {"output": {"path": str(tmp_path)}, "sample": {"gallery": {"count": 2}}}

    gallery = Gallery.from_config(config, {"ann": "woman", "bob": "man"})

    assert (gallery.enabled, gallery.count, gallery.people) == (True, 2, {"ann": "woman", "bob": "man"})
    assert gallery.dir == tmp_path / "samples" / "gallery"


def test_a_class_word_and_the_dataset_trigger_name_the_one_person_when_there_are_no_classes(tmp_path: Path) -> None:
    config = {"prior": {"class_word": "woman"}, "dataset": {"trigger": "ann"}, "output": {"path": str(tmp_path)}}

    assert Gallery.from_config(config, {}).people == {"ann": "woman"}


@pytest.mark.parametrize("config", [{}, {"prior": {"weight": 0.0}}, {"prior": None}])
def test_a_config_with_nobody_to_show_has_no_gallery(config: dict[str, object], tmp_path: Path) -> None:
    assert not Gallery.from_config({**config, "output": {"path": str(tmp_path)}}, {}).enabled
