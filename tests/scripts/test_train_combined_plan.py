"""scripts/train_combined.py: what the plan file and the member configs must contain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import train_combined
from tests.scripts.combined_helpers import write_plan


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trainer": None}, "needs 'trainer:' naming the trainer script"),
        ({"trainer": 5}, "needs 'trainer:' naming the trainer script"),
        ({"members": None}, "needs a 'members:' list of at least two configs"),
        ({"members": [{"config": "a.yaml"}]}, "needs a 'members:' list of at least two configs"),
        ({"members": [{"config": "a.yaml"}, "b.yaml"]}, r"members\[1\] needs a 'config'"),
        ({"members": [{"config": "a.yaml"}, {"strength": 1}]}, r"members\[1\] needs a 'config'"),
        (
            {"members": [{"config": "a.yaml", "strength": 0}, {"config": "b.yaml"}]},
            r"members\[0\] strength must be above 0, got 0",
        ),
        ({"output": None}, "needs 'output:' with 'path' and 'name'"),
        ({"output": {"path": "x"}}, "needs 'output:' with 'path' and 'name'"),
        ({"output": {"path": "x", "name": "y", "dtype": "int4"}}, r"output dtype must be one of \[.*\], got 'int4'"),
    ],
)
def test_a_plan_that_is_incomplete_is_refused_with_the_reason(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    plan = write_plan(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        train_combined.load_plan(plan)


def test_a_valid_plan_gives_the_trainer_members_with_default_strength_output_file_and_dtype(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, members=[{"config": "a.yaml"}, {"config": "b.yaml", "strength": "0.25"}])

    trainer, members, output, dtype = train_combined.load_plan(plan)

    assert trainer == tmp_path / "trainer.py"
    assert members == [(Path("a.yaml"), 1.0), (Path("b.yaml"), 0.25)]
    assert (output, dtype) == (tmp_path / "soup" / "soup.safetensors", "float32")


@pytest.mark.parametrize(("text", "message"), [("- a\n- b\n", "not a YAML mapping"), ("", "needs 'trainer:'")])
def test_a_plan_file_must_be_a_mapping_and_an_empty_one_lacks_the_trainer(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        train_combined.load_plan(path)


def test_non_breaking_spaces_in_a_yaml_file_count_as_spaces(tmp_path: Path) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(
        "trainer: t.py\nmembers: [{config: a}, {config: b}]\noutput: {path: o, name: n}\n", encoding="utf-8"
    )

    assert train_combined.load_plan(path)[0] == Path("t.py")


@pytest.mark.parametrize("name", ["lora", "lora.safetensors"])
def test_the_member_output_is_the_path_and_name_with_the_safetensors_extension(name: str) -> None:
    assert train_combined.member_output({"path": "out", "name": name}) == Path("out") / "lora.safetensors"


def test_a_member_config_without_an_output_cannot_say_where_its_lora_goes(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("output: {path: out}\n")

    with pytest.raises(ValueError, match=r"has no output\.path / output\.name"):
        train_combined.trained_lora(path)
