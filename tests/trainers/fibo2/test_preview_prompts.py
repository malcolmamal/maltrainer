"""More prompts for g, and the progress bar of a render."""

from __future__ import annotations

from typing import Any

import pytest

from maltrainer.trainers.fibo2 import previews

from .runs import prepared
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

    assert set(trainer.preview_keys()) == {"g"} and trainer.preview_texts() == {"a cat", ""}


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
