"""More prompts for g, the settings of the sample section, and the progress bar of a render."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from maltrainer.trainers.visage import previews

from .runs import prepared, run_lora
from .test_previews import SAMPLE, _trainer, decoder  # noqa: F401 (the autouse decoder fixture)
from .world import Workspace


def test_more_prompts_for_g_are_listed_encoded_and_rendered_as_p1_p2(workspace: Workspace) -> None:
    workspace.add_images(1)
    sample = {**SAMPLE, "prompts": ["a woman reading in a cafe", "a man on a beach"]}
    trainer = prepared(workspace.config(sample=sample))

    keys = trainer.preview_keys()
    keys["g"][1]()

    assert set(keys) == {"w", "m", "g"} and keys["g"][0] == "render every preview prompt (4)"
    assert {"a woman reading in a cafe", "a man on a beach"} <= set(trainer.text_cache)
    names = sorted(p.name for p in (workspace.out / "samples").iterdir())
    assert names == ["step000000_m.png", "step000000_p1.png", "step000000_p2.png", "step000000_w.png"]


def test_a_prompts_only_sample_section_still_gets_g(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(sample={"prompts": ["a cat"], "resolution": 32, "steps": 2}))

    assert set(trainer.preview_keys()) == {"g"} and trainer.preview_texts() == {"a cat"}


def _record_sampler(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """sample_latents replaced by a stub that keeps its arguments after the transformer and the caption."""
    seen: list[tuple[Any, ...]] = []

    def stub(_transformer: Any, _cond: torch.Tensor, *args: Any) -> torch.Tensor:
        seen.append(args)
        return torch.zeros(1, 8, 2, 2)

    monkeypatch.setattr(previews, "sample_latents", stub)
    return seen


def test_the_sampler_settings_of_the_sample_section_reach_the_sampler(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_sampler(monkeypatch)
    sample = {**SAMPLE, "seed": 9, "steps": 3, "guidance": 2.0, "solver": "euler", "churn": 0.1, "shift": 2.0}
    trainer = _trainer(workspace, sample=sample)

    trainer.preview_keys()["w"][1]()

    grid, seed, device, dtype, steps, guidance, solver, churn, shift, _on_step = seen[0]
    assert (grid, seed, steps, guidance, solver, churn, shift) == ((2, 2), 9, 3, 2.0, "euler", 0.1, 2.0)
    assert (device, dtype) == (trainer.device, trainer.dtype)


def test_the_defaults_are_the_released_sampling_recipe(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_sampler(monkeypatch)
    workspace.add_images(1)
    trainer = prepared(workspace.config(sample={"keys": {"w": "photo"}}))

    trainer.preview_keys()["w"][1]()

    grid, seed, _device, _dtype, steps, guidance, solver, churn, shift, _on_step = seen[0]
    assert (grid, seed, steps, guidance, solver, churn, shift) == ((2, 2), 42, 50, 3.0, "sde", 0.3, 3.3)


def test_a_progress_bar_counts_the_sampling_steps_of_each_image(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    bars: list[dict[str, Any]] = []

    class Bar:
        def __init__(self, total: int, desc: str, leave: bool) -> None:
            self.n = 0
            bars.append({"total": total, "desc": desc, "leave": leave, "final": 0})
            self.record = bars[-1]

        def __enter__(self) -> Bar:
            return self

        def __exit__(self, *_: object) -> None:
            self.record["final"] = self.n

        def update(self, count: int) -> None:
            self.n += count

    monkeypatch.setattr(previews, "tqdm", Bar)
    trainer = _trainer(workspace)

    trainer.preview_keys()["g"][1]()

    assert [(b["desc"], b["total"], b["final"], b["leave"]) for b in bars] == [
        ("Sampling (w)", 2, 2, False),
        ("Sampling (m)", 2, 2, False),
    ]


def test_the_step_the_images_are_named_by_follows_the_run(workspace: Workspace) -> None:
    workspace.add_images(2)
    trainer = run_lora(workspace.config(sample=SAMPLE))

    assert trainer.current_step == 3


def test_a_press_is_logged_with_its_prompt_and_path(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    trainer = _trainer(workspace)

    with caplog.at_level(logging.INFO):
        trainer.preview_keys()["m"][1]()

    assert 'Preview [m] "photo of a man" ->' in caplog.text and "step000000_m.png" in caplog.text
