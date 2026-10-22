"""scripts/train_combined.py: members train one after another and the soup is merged at the end."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

import train_combined
from tests.scripts.combined_helpers import D1, D2, U1, U2, Trainers, write_plan
from tests.scripts.helpers import delta, write_lora


@pytest.fixture
def trainers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Trainers:
    fake = Trainers()
    monkeypatch.setattr(train_combined, "subprocess", SimpleNamespace(call=fake.call))
    monkeypatch.chdir(tmp_path)
    return fake


def _run(monkeypatch: pytest.MonkeyPatch, *argv: object) -> int:
    monkeypatch.setattr("sys.argv", ["train_combined.py", *map(str, argv)])
    return train_combined.main()


def test_each_member_trains_in_turn_and_the_soup_applies_their_strengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path)

    assert _run(monkeypatch, "--config", plan) == 0

    assert [c[1:] for c in trainers.commands] == [
        [str(tmp_path / "trainer.py"), "--config", str(tmp_path / "first.yaml")],
        [str(tmp_path / "trainer.py"), "--config", str(tmp_path / "second.yaml")],
    ]
    merged = load_file(str(tmp_path / "soup" / "soup.safetensors"))
    total = merged["m.lora_B.weight"].double() @ merged["m.lora_A.weight"].double()
    assert torch.allclose(total, 0.5 * delta(D1, U1, 2.0) + 2 * delta(D2, U2, 4.0), atol=1e-5)
    assert merged["m.lora_A.weight"].shape == (4, 4) and merged["m.lora_A.weight"].dtype == torch.float32
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == f"Merged LoRA: {(tmp_path / 'soup' / 'soup.safetensors').resolve()}"
    assert "[combined] 1 modules at rank [4]" in lines


def test_arguments_after_the_double_dash_go_to_every_trainer_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers
) -> None:
    plan = write_plan(
        tmp_path, output={"path": str(tmp_path / "soup"), "name": "soup.safetensors", "dtype": "bfloat16"}
    )

    assert _run(monkeypatch, "--config", plan, "--", "--dry-run") == 0

    assert [c[-1] for c in trainers.commands] == ["--dry-run", "--dry-run"]
    assert load_file(str(tmp_path / "soup" / "soup.safetensors"))["m.lora_A.weight"].dtype == torch.bfloat16


def test_a_trainer_script_given_relative_to_the_repo_root_is_found_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers
) -> None:
    plan = write_plan(tmp_path, trainer="train_moss_image_lora.py")

    assert _run(monkeypatch, "--config", plan) == 0

    assert trainers.commands[0][1] == str(train_combined.ROOT / "train_moss_image_lora.py")


def test_a_member_that_fails_stops_the_run_and_nothing_is_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers
) -> None:
    trainers.codes = [3]

    with pytest.raises(SystemExit, match=r"first\.yaml failed \(exit code 3\); nothing merged"):
        _run(monkeypatch, "--config", write_plan(tmp_path))

    assert len(trainers.commands) == 1 and not (tmp_path / "soup").exists()


def test_a_member_that_exits_cleanly_without_writing_its_lora_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers
) -> None:
    trainers.skip_writing = True

    with pytest.raises(SystemExit, match=r"first\.yaml finished but wrote no new .*lora_first\.safetensors"):
        _run(monkeypatch, "--config", write_plan(tmp_path))


def test_a_lora_left_over_from_an_earlier_run_does_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers
) -> None:
    plan = write_plan(tmp_path)
    (tmp_path / "first").mkdir()
    old = write_lora(tmp_path / "first" / "lora_first.safetensors", {"m": (D1, U1, 2.0)})
    os.utime(old, (1_000_000, 1_000_000))
    trainers.skip_writing = True

    with pytest.raises(SystemExit, match="wrote no new"):
        _run(monkeypatch, "--config", plan)


def test_a_merge_that_cannot_be_made_is_reported_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers, capsys: pytest.CaptureFixture[str]
) -> None:
    trainers.lora = {"m": (torch.ones(2, 5), torch.ones(3, 2), 2.0)}

    assert _run(monkeypatch, "--config", write_plan(tmp_path)) == 1

    assert "[combined] merge failed: m: inputs disagree on the layer shape" in capsys.readouterr().err
    assert not (tmp_path / "soup").exists()


def test_an_existing_merged_file_is_kept_unless_force_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainers: Trainers, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path)
    (tmp_path / "soup").mkdir()
    (tmp_path / "soup" / "soup.safetensors").write_bytes(b"old")

    with pytest.raises(SystemExit):
        _run(monkeypatch, "--config", plan)
    assert "soup.safetensors exists; pass --force to overwrite" in capsys.readouterr().err
    assert trainers.commands == [] and _run(monkeypatch, "--config", plan, "--force") == 0

    assert "m.lora_A.weight" in load_file(str(tmp_path / "soup" / "soup.safetensors"))


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        ("member", "member config not found: "),
        ("trainer", "trainer script not found: "),
        ("plan", "needs 'trainer:'"),
    ],
)
def test_a_plan_whose_files_are_missing_is_refused_before_any_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainers: Trainers,
    capsys: pytest.CaptureFixture[str],
    break_it: str,
    message: str,
) -> None:
    plan = write_plan(tmp_path)
    if break_it == "member":
        (tmp_path / "second.yaml").unlink()
    elif break_it == "trainer":
        (tmp_path / "trainer.py").unlink()
    else:
        plan.write_text("members: []\n")

    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--config", plan)

    assert stop.value.code == 2 and message in capsys.readouterr().err
    assert trainers.commands == []
