"""Previews rendered with the weights in training: when, which rows, and that training is left untouched."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import train_moss_image_finetune as finetune

from .fakes import Fakes
from .runs import run_finetune, same_values
from .world import SAMPLE, Workspace, named_weights


def _files(ws: Workspace) -> list[str]:
    return sorted(path.name for path in (ws.out / "samples").iterdir())


def test_previews_render_before_the_first_step_every_n_steps_and_at_the_end(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)

    run_finetune(workspace.config(sample=SAMPLE, train={"max_steps": 3}))

    assert _files(workspace) == [
        "prompts.txt",
        "step000000.png",
        "step000000_00.png",
        "step000002.png",
        "step000002_00.png",
        "step000003.png",
        "step000003_00.png",
    ]
    listing = (workspace.out / "samples" / "prompts.txt").read_text(encoding="utf-8")
    assert "00  seed 7  [on ]  a photo of zed" in listing and "01  seed 8  [off]  a photo of a person" in listing
    assert fakes.decoder_loads == 1


def test_with_ema_each_step_renders_a_raw_row_and_an_ema_row(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    run_finetune(workspace.config(sample={**SAMPLE, "every": 100}, train={"ema_decay": 0.5, "max_steps": 1}))

    assert _files(workspace) == [
        "prompts.txt",
        "step000000.png",
        "step000000_00_ema.png",
        "step000000_00_raw.png",
        "step000001.png",
        "step000001_00_ema.png",
        "step000001_00_raw.png",
    ]


def test_previews_leave_the_training_result_bit_for_bit_unchanged(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    ema = {"ema_decay": 0.5, "ema_warmup": False}
    quiet = run_finetune(workspace.config(output={"path": str(workspace.root / "quiet")}, train=ema))

    shown = run_finetune(workspace.config(sample={**SAMPLE, "every": 1}, train=ema))

    assert shown.ema is not None and quiet.ema is not None
    assert same_values(named_weights(shown.transformer), named_weights(quiet.transformer))
    assert same_values(shown.ema.averaged(), quiet.ema.averaged())
    assert shown.transformer.training


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["raw"], ["step000000_00_raw.png", "step000001_00_raw.png"]),
        (["ema"], ["step000000_00_ema.png", "step000001_00_ema.png"]),
        ([], []),
    ],
)
def test_the_rows_setting_picks_which_weights_render(
    workspace: Workspace, rows: list[str], expected: list[str]
) -> None:
    workspace.add_images("p", 1)

    run_finetune(
        workspace.config(sample={**SAMPLE, "every": 100, "rows": rows}, train={"ema_decay": 0.5, "max_steps": 1})
    )

    assert [name for name in _files(workspace) if name.endswith(("raw.png", "ema.png"))] == expected


def test_a_resumed_run_does_not_render_the_baseline_again(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    sample: dict[str, Any] = {**SAMPLE, "every": 10}
    run_finetune(workspace.config(sample=sample, train={"max_steps": 2}))
    before = Path(workspace.out / "samples" / "step000000.png").stat().st_mtime_ns
    Path(workspace.out / "samples" / "step000000.png").unlink()

    run_finetune(
        workspace.config(sample=sample, train={"max_steps": 3, "resume_from": str(workspace.out / "ft_state")})
    )

    assert before and not (workspace.out / "samples" / "step000000.png").exists()
    assert (workspace.out / "samples" / "step000003.png").exists()


def test_a_prompt_that_is_not_encoded_yet_is_skipped_for_the_round(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 1)
    config = workspace.config(sample={**SAMPLE, "every": 100}, train={"max_steps": 1})

    trainer = finetune.MOSSImageFinetuneTrainer(config)
    monkeypatch.setattr(trainer, "_preview_ready", lambda _prompt: False)
    trainer.run()

    assert [name for name in _files(workspace) if name.endswith(".png")] == []
