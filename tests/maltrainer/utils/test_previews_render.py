from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Tuple

import pytest
from PIL import Image

from maltrainer.utils import previews as previews_module
from maltrainer.utils.previews import Previews

COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


class _Renderer:
    """Paints each image a colour in call order; every image takes ``seconds`` on the fake clock."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.now = 0.0
        self.seconds = seconds
        self.calls: List[Tuple[str, int]] = []

    def time(self) -> float:
        return self.now

    def __call__(self, prompt: str, seed: int) -> Image.Image:
        self.now += self.seconds
        colour = COLOURS[len(self.calls)]
        self.calls.append((prompt, seed))
        return Image.new("RGB", (4, 2), colour)


@pytest.fixture
def renderer(monkeypatch: pytest.MonkeyPatch) -> _Renderer:
    fake = _Renderer()
    monkeypatch.setattr(previews_module, "time", SimpleNamespace(time=fake.time))
    return fake


def _previews(tmp_path: Path, **config: object) -> Previews:
    return Previews(
        {"every": 100, "seed": 40, "prompts": ["a", {"prompt": "b", "enabled": False}, "c"], **config}, tmp_path
    )


def test_each_enabled_prompt_renders_with_its_slot_seed_into_a_strip(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path)

    previews.render(100, renderer)

    samples = tmp_path / "samples"
    assert renderer.calls == [("a", 40), ("c", 42)]
    assert sorted(path.name for path in samples.iterdir()) == [
        "prompts.txt",
        "step000100.png",
        "step000100_00.png",
        "step000100_02.png",
    ]
    with Image.open(samples / "step000100.png") as strip:
        assert strip.size == (8, 2)
        assert [strip.getpixel((0, 0)), strip.getpixel((4, 1))] == [COLOURS[0], COLOURS[1]]
    assert previews.current == "step 100, image 2/2"


def test_each_variant_renders_in_its_own_weights_as_a_row(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path)
    entered: List[str] = []

    @contextlib.contextmanager
    def ema_weights() -> Iterator[None]:
        entered.append(f"in after {len(renderer.calls)}")
        yield
        entered.append(f"out after {len(renderer.calls)}")

    previews.render(300, renderer, variants=[("raw", contextlib.nullcontext), ("ema", ema_weights)])

    samples = tmp_path / "samples"
    assert entered == ["in after 2", "out after 4"]
    assert {"step000300_00_raw.png", "step000300_02_ema.png"} <= {path.name for path in samples.iterdir()}
    with Image.open(samples / "step000300.png") as strip:
        assert strip.size == (8, 4)
        assert [strip.getpixel((0, 0)), strip.getpixel((4, 3))] == [COLOURS[0], COLOURS[3]]
    assert previews.current == "step 300, image 4/4, ema"


def test_a_variant_whose_row_is_off_is_not_rendered(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path, rows=["raw"])

    previews.render(300, renderer, variants=[("raw", contextlib.nullcontext), ("ema", contextlib.nullcontext)])

    assert len(renderer.calls) == 2
    assert not (tmp_path / "samples" / "step000300_00_ema.png").exists()


def test_with_every_row_off_nothing_renders(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path, rows=[])

    seconds = previews.render(300, renderer, variants=[("raw", contextlib.nullcontext)])

    assert seconds == 0.0
    assert renderer.calls == []
    assert [path.name for path in (tmp_path / "samples").iterdir()] == ["prompts.txt"]


def test_prompts_still_being_encoded_are_skipped_and_logged(
    tmp_path: Path, renderer: _Renderer, caplog: pytest.LogCaptureFixture
) -> None:
    previews = _previews(tmp_path)

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.previews"):
        previews.render(100, renderer, ready=lambda prompt: prompt != "c")

    assert renderer.calls == [("a", 40)]
    assert caplog.messages[0] == "[sample] step 100: 1 new prompt(s) still being encoded, skipped this round: [02]"


def test_nothing_renders_while_no_prompt_is_ready(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path)

    assert previews.render(100, renderer, ready=lambda _prompt: False) == 0.0
    assert not (tmp_path / "samples" / "step000100.png").exists()


def test_the_render_time_is_returned_and_slow_images_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slow = _Renderer(seconds=30.0)
    monkeypatch.setattr(previews_module, "time", SimpleNamespace(time=slow.time))
    previews = _previews(tmp_path, slow_after=20)

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.previews"):
        seconds = previews.render(100, slow)

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert seconds == 60.0
    assert len(warnings) == 2
    assert warnings[0].startswith("[sample] step 100, image 1/2 took 30s (over 20s): the GPU is probably out of VRAM")
    assert caplog.messages[-1].startswith("[sample] step 100: 2 preview(s) in 60s -> ")


def test_the_listing_is_written_once_per_process(tmp_path: Path, renderer: _Renderer) -> None:
    previews = _previews(tmp_path)
    previews.render(100, renderer, ready=lambda _prompt: False)
    listing = tmp_path / "samples" / "prompts.txt"
    listing.write_text("edited\n", encoding="utf-8")

    previews.render(200, renderer, ready=lambda _prompt: False)

    assert listing.read_text(encoding="utf-8") == "edited\n"
