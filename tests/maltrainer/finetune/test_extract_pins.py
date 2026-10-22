"""Boundaries and averages of the LoRA extraction that the other extract tests leave open (mutation testing)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.finetune.extract import GENERIC, extract, factor, keep_count


@pytest.mark.parametrize(
    ("s", "mode", "value", "expected"),
    [
        ([0.8, 0.5, 0.1], "ratio", 0.1, 3),  # a largest singular value of at most 1 is not "no energy"
        ([2.0, 0.0, 0.0], "ratio", 0.0, 3),  # a zero second value is not "no energy" either
        ([4.0, 3.0], "fro", 0.67, 2),  # energy is the square: 16 / 25 = 0.64 < 0.67 (a cube would give 0.70)
        ([3.0, 2.0, 1.0], "threshold", 2.0, 2),  # a value exactly at the threshold is kept
    ],
)
def test_keep_count_boundaries(s: list[float], mode: str, value: float, expected: int) -> None:
    assert keep_count(torch.tensor(s), mode, value, 3) == expected


def test_a_small_difference_still_reports_the_share_it_dropped() -> None:
    diff = torch.tensor([[0.5, 0.0], [0.0, 0.1]])  # ||diff||^2 = 0.26, below 1

    _, _, k, error = factor(diff, [1], "fixed", 0.0, "cpu")[1]

    assert k == 1
    assert error == pytest.approx((0.01 / 0.26) ** 0.5)


def test_the_profile_is_frozen_and_so_hashable() -> None:
    assert hash(GENERIC) == hash(GENERIC)


def test_the_unknown_mode_error_lists_the_modes(tmp_path: Path) -> None:
    save_file({"a.weight": torch.ones(2, 2)}, str(tmp_path / "w.safetensors"))
    path = tmp_path / "w.safetensors"

    with pytest.raises(ValueError, match=r"\('fixed', 'fro', 'ratio', 'threshold'\)"):
        extract(path, path, [1], mode="quantile")


def test_the_report_averages_the_ranks_and_dropped_shares_over_the_modules(tmp_path: Path) -> None:
    base = {"a.weight": torch.zeros(2, 3), "c.weight": torch.zeros(2, 3)}
    tuned = {
        "a.weight": torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # singular values 3, 1: drops 1 / 10
        "c.weight": torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),  # 2, 0.5: drops 0.25 / 4.25
    }
    save_file(base, str(tmp_path / "base.safetensors"))
    save_file(tuned, str(tmp_path / "tuned.safetensors"))

    _, report = extract(tmp_path / "base.safetensors", tmp_path / "tuned.safetensors", [2], "threshold", 2.0)

    per_cap = report["per_cap"][2]
    first, second = (1 / 10) ** 0.5, (0.25 / 4.25) ** 0.5
    assert (per_cap["modules"], per_cap["rank_min"], per_cap["rank_max"], per_cap["rank_mean"]) == (2, 1, 1, 1.0)
    assert per_cap["dropped_mean"] == pytest.approx((first + second) / 2)
    assert per_cap["dropped_worst"] == pytest.approx(first)
