"""scripts/qwen_2_1_stack_probe.py main(): the frozen-model loss table against its two baselines (CPU fakes)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch

import qwen_2_1_stack_probe as probe
from tests.scripts.qwen_fakes import FakeTransformer, install, write_config
from tests.scripts.world_images import write_dataset


class Silent(FakeTransformer):
    """A model that predicts nothing: its loss is the zero baseline."""

    def forward(self, hidden_states: torch.Tensor, **_inputs: Any) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def _probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *flags: str,
    images: int = 3,
    model: type[FakeTransformer] | None = None,
) -> None:
    install(monkeypatch, probe)
    if model:
        monkeypatch.setattr(probe, "load_qwen_image_2_1_transformer", lambda _path, **_options: model())
    config = write_config(tmp_path / "config.yaml", tmp_path / "out", write_dataset(tmp_path / "data", images))
    monkeypatch.setattr("sys.argv", ["probe", "--config", str(config), "--repeats", "2", *flags])
    probe.main()


def _rows(out: str) -> list[list[float]]:
    return [[float(v) for v in line.split()] for line in out.splitlines() if re.match(r"\s+0\.\d\d ", line)]


def test_one_row_per_sigma_with_the_matched_and_mismatched_loss_and_both_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _probe(monkeypatch, tmp_path)

    out = capsys.readouterr().out
    rows = _rows(out)
    assert [r[0] for r in rows] == [0.15, 0.3, 0.5, 0.7, 0.9]
    assert all(len(r) == 5 and all(v > 0 for v in r[1:]) for r in rows)
    assert "probing 3 images at 64x64" in out and "latent std " in out
    assert "PASS if `matched` sits well under" in out


def test_a_model_that_predicts_nothing_scores_the_zero_baseline_with_or_without_the_right_caption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _probe(monkeypatch, tmp_path, model=Silent)

    for sigma, matched, mismatched, zero, noise in _rows(capsys.readouterr().out):
        assert matched == pytest.approx(zero, rel=1e-3) and mismatched == pytest.approx(zero, rel=1e-3), sigma
        assert noise > 0


def test_only_as_many_images_as_asked_for_are_probed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _probe(monkeypatch, tmp_path, "--samples", "2", images=4)

    assert "probing 2 images at" in capsys.readouterr().out
