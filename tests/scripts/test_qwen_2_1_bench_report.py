"""scripts/qwen_2_1_bench_report.py: log parsing, adapter magnitude and the report it writes (plots are mocked)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.scripts.bench_report_world import fake_matplotlib, make_bench, write_adapter, write_log


@pytest.fixture
def pyplot(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[MagicMock, ModuleType]]:
    with fake_matplotlib(monkeypatch) as loaded:
        yield loaded


@pytest.fixture
def report(pyplot: tuple[MagicMock, ModuleType]) -> ModuleType:
    return pyplot[1]


def test_a_log_gives_steps_losses_learning_rates_and_the_speed_between_first_and_last_line(
    report: ModuleType, tmp_path: Path
) -> None:
    parsed = report.parse_log(write_log(tmp_path / "training.log", [1.0, 0.5, 0.25]))

    assert (parsed["step"], parsed["loss"]) == ([1, 2, 3], [1.0, 0.5, 0.25])
    assert parsed["lr"] == pytest.approx([1e-3, 5e-4, 3.33e-4], rel=1e-2)
    assert parsed["sec_per_step"] == 15.0 and parsed["wall_minutes"] == 0.5


def test_a_log_without_progress_lines_is_refused(report: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "training.log").write_text("hello\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"no progress lines in .*training\.log"):
        report.parse_log(tmp_path / "training.log")


def test_the_loss_average_starts_at_the_first_value_and_follows_with_the_given_weight(report: ModuleType) -> None:
    assert report.ema([4.0, 4.0, 4.0]).tolist() == [4.0, 4.0, 4.0]
    assert report.ema([0.0, 10.0], alpha=0.5).tolist() == [0.0, 5.0]
    assert report.ema([1.0, 2.0])[1] == pytest.approx(0.02 * 2 + 0.98 * 1)
    assert isinstance(report.ema([1.0]), np.ndarray)


def test_the_adapter_magnitude_is_the_frobenius_norm_of_b_a_times_alpha_over_rank_averaged_over_layers(
    report: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "a.safetensors"
    save_file(
        {
            "m.lora_A.weight": torch.eye(2),
            "m.lora_B.weight": torch.diag(torch.tensor([3.0, 4.0])),
            "m.alpha": torch.tensor(4.0),
            "n.lora_A.weight": torch.tensor([[1.0, 2.0, 2.0]]),
            "n.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(path),
    )

    assert report.adapter_norm(path) == pytest.approx((5 * 4 / 2 + 15.0 * 1) / 2)


def test_a_larger_adapter_has_a_larger_magnitude(report: ModuleType, tmp_path: Path) -> None:
    small, big = write_adapter(tmp_path / "s.safetensors", 1.0), write_adapter(tmp_path / "b.safetensors", 3.0)

    assert report.adapter_norm(big) == pytest.approx(3 * report.adapter_norm(small))


@pytest.mark.parametrize(
    ("name", "step"),
    [("run_step_600.safetensors", "600"), ("run.safetensors", "final"), ("run_step_a.safetensors", "final")],
)
def test_checkpoint_steps_come_from_the_file_name_and_final_takes_the_last_logged_step(
    report: ModuleType, name: str, step: str
) -> None:
    assert report.checkpoint_step(Path(name)) == step
    assert report.step_num(step, 2000) == (2000 if step == "final" else int(step))


@pytest.mark.parametrize(
    ("name", "expected"),
    [("adamw", "#2f6fdf"), ("rhea_prodigy", "#d9480f"), ("automagic3", "#2b8a3e"), ("base", "#868e96"), ("lion", None)],
)
def test_runs_are_coloured_by_their_optimizer_even_with_a_prefix(
    report: ModuleType, name: str, expected: str | None
) -> None:
    assert report.color(name) == expected


def test_main_writes_the_report_json_the_grids_and_both_charts(
    pyplot: tuple[MagicMock, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plt, report = pyplot
    bench = make_bench(tmp_path)
    runs = [f"adamw={tmp_path / 'out_adamw'}", f"prodigy={tmp_path / 'out_prodigy'}"]
    monkeypatch.setattr("sys.argv", ["report", "--bench", str(bench), "--runs", *runs, "--grid-steps", "5", "final"])

    report.main()

    out = bench / "report"
    saved = json.loads((out / "report.json").read_text())
    adamw = saved["dynamics"]["adamw"]
    assert set(saved["dynamics"]) == {"adamw", "prodigy"}
    assert set(adamw) >= {"sec_per_step", "wall_minutes", "loss_first200", "loss_last200", "lr_final", "lr_max", "dw"}
    assert "loss" not in adamw and adamw["loss_first200"] == pytest.approx(0.58333, rel=1e-3)
    assert adamw["lr_final"] == pytest.approx(1e-3 / 3, rel=1e-2) and adamw["lr_max"] == 1e-3
    assert [step for step, _ in adamw["dw"]] == [5, 3]
    assert adamw["dw"][1][1] == pytest.approx(2 * adamw["dw"][0][1])
    assert saved["summary"]["reference"] == {"training_pairwise_clip": 0.7}
    assert sorted(p.name for p in out.glob("grid_*.png")) == ["grid_knight.png", "grid_portrait.png"]
    charts = [fig.savefig.call_args.args[0] for fig in plt.figures]
    assert charts == [out / "training_dynamics.png", out / "image_metrics.png"]
    assert "adamw:" in capsys.readouterr().out


def test_a_metric_that_was_not_scored_is_left_out_of_its_chart_and_so_is_its_base_reference_line(
    pyplot: tuple[MagicMock, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plt, report = pyplot
    bench = make_bench(tmp_path, runs=("adamw",))
    summary = json.loads((bench / "summary.json").read_text())
    for row in summary["rows"]:
        row["adherence"] = None
        if row["run"] == "base":
            row["diversity"] = row["memorization"] = None
    (bench / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr("sys.argv", ["report", "--bench", str(bench), "--runs", f"adamw={tmp_path / 'out_adamw'}"])

    report.main()

    likeness, adherence, diversity, memorization = plt.axes_by_figure[1]
    assert likeness.plot.call_count == 1 and adherence.plot.call_count == 0 and diversity.plot.call_count == 1
    assert [c.args[0] for c in likeness.axhline.call_args_list] == [0, 1]
    assert diversity.axhline.call_count == 0
    assert [c.args[0] for c in memorization.axhline.call_args_list] == [0.7]
