"""LoKr modules in scripts/merge_lora_into_model.py, applied as ComfyUI's LoKrAdapter.calculate_weight applies them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.scripts.merge_into_model_world import P, model_file, run

M = "diffusion_model.blocks.0.lin"
GEN = torch.Generator().manual_seed(9)


def _lokr(path: Path, **parts: torch.Tensor) -> Path:
    save_file({f"{M}.{name}": tensor for name, tensor in parts.items()}, str(path))
    return path


def _merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lokr: Path, strength: float
) -> tuple[torch.Tensor, torch.Tensor]:
    base, out = model_file(tmp_path / "base.safetensors", 1), tmp_path / "out.safetensors"
    assert run(monkeypatch, str(base), str(out), f"{lokr}:{strength}") == 0
    key = f"{P}blocks.0.lin.weight"
    return load_file(str(base))[key], load_file(str(out))[key]


def test_whole_factors_add_strength_times_their_kronecker_product_and_alpha_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ComfyUI: with lokr_w1 and lokr_w2 both stored whole there is no rank, so the scale is 1.0 whatever alpha says
    w1, w2 = torch.randn(2, 2, generator=GEN), torch.randn(2, 3, generator=GEN)
    lokr = _lokr(tmp_path / "lokr.safetensors", lokr_w1=w1, lokr_w2=w2, alpha=torch.tensor(8.0))

    base, merged = _merged(tmp_path, monkeypatch, lokr, 0.5)

    torch.testing.assert_close(merged, base + 0.5 * torch.kron(w1, w2))


def test_a_low_rank_w2_is_its_pair_product_and_scales_by_alpha_over_its_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w1 = torch.randn(2, 2, generator=GEN)
    a, b = torch.randn(2, 1, generator=GEN), torch.randn(1, 3, generator=GEN)  # rank = b's rows = 1
    lokr = _lokr(tmp_path / "lokr.safetensors", lokr_w1=w1, lokr_w2_a=a, lokr_w2_b=b, alpha=torch.tensor(0.5))

    base, merged = _merged(tmp_path, monkeypatch, lokr, 2.0)

    torch.testing.assert_close(merged, base + 2.0 * (0.5 / 1) * torch.kron(w1, a @ b))


def test_with_both_factors_low_rank_the_scale_uses_w2s_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w1a, w1b = torch.randn(2, 3, generator=GEN), torch.randn(3, 2, generator=GEN)  # w1 rank 3
    w2a, w2b = torch.randn(2, 2, generator=GEN), torch.randn(2, 3, generator=GEN)  # w2 rank 2: ComfyUI's dim
    lokr = _lokr(
        tmp_path / "lokr.safetensors",
        lokr_w1_a=w1a,
        lokr_w1_b=w1b,
        lokr_w2_a=w2a,
        lokr_w2_b=w2b,
        alpha=torch.tensor(4.0),
    )

    base, merged = _merged(tmp_path, monkeypatch, lokr, 1.0)

    torch.testing.assert_close(merged, base + (4.0 / 2) * torch.kron(w1a @ w1b, w2a @ w2b))


@pytest.mark.parametrize(
    "parts, message",
    [
        (
            {"lokr_w1": torch.ones(2, 2), "lokr_w2": torch.ones(2, 3), "lokr_t2": torch.ones(1, 1, 1, 1)},
            "convolutional or DoRA",
        ),
        (
            {"lokr_w1": torch.ones(2, 2), "lokr_w2_a": torch.ones(2, 1)},
            "neither lokr_w2 nor both lokr_w2_a and lokr_w2_b",
        ),
        (
            {"lokr_w1": torch.ones(3, 3), "lokr_w2": torch.ones(2, 3)},
            r"the LoKr product is \(6, 9\), the base weight \(4, 6\)",
        ),
    ],
)
def test_lokr_it_cannot_apply_exactly_is_refused_and_nothing_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    parts: dict[str, torch.Tensor],
    message: str,
) -> None:
    base, out = model_file(tmp_path / "base.safetensors", 1), tmp_path / "out.safetensors"
    assert run(monkeypatch, str(base), str(out), str(_lokr(tmp_path / "lokr.safetensors", **parts))) == 1
    assert re.search(message, capsys.readouterr().err) and not out.exists()
