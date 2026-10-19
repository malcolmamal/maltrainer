from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.finetune.extract import GENERIC, ExtractProfile, extract

DIFF = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def _base() -> Dict[str, torch.Tensor]:
    return {
        "a.weight": torch.ones(2, 3),
        "a.bias": torch.zeros(2),
        "norm.weight": torch.ones(3),
        "b.weight": torch.full((2, 2), 0.5),
    }


def _tuned() -> Dict[str, torch.Tensor]:
    tuned = _base()
    tuned["a.weight"] = tuned["a.weight"] + DIFF
    tuned["a.bias"] = torch.ones(2)
    tuned["norm.weight"] = torch.full((3,), 2.0)
    return tuned


def _files(tmp_path: Path, base: Dict[str, torch.Tensor], tuned: Dict[str, torch.Tensor]) -> Tuple[Path, Path]:
    base_path, tuned_path = tmp_path / "base.safetensors", tmp_path / "tuned.safetensors"
    save_file(base, str(base_path))
    save_file(tuned, str(tuned_path))
    return base_path, tuned_path


def test_a_changed_linear_becomes_a_lora_that_rebuilds_its_difference(tmp_path: Path) -> None:
    base, tuned = _files(tmp_path, _base(), {f"model.diffusion_model.{k}": v for k, v in _tuned().items()})

    states, _ = extract(base, tuned, [2], dtype=torch.float32)

    lora = states[2]
    assert sorted(lora) == [
        "diffusion_model.a.alpha",
        "diffusion_model.a.lora_A.weight",
        "diffusion_model.a.lora_B.weight",
    ]
    torch.testing.assert_close(lora["diffusion_model.a.lora_B.weight"] @ lora["diffusion_model.a.lora_A.weight"], DIFF)
    assert lora["diffusion_model.a.alpha"].dtype == torch.float32
    assert float(lora["diffusion_model.a.alpha"]) == 2.0


def test_the_report_counts_layers_and_what_each_cap_dropped(tmp_path: Path) -> None:
    base, tuned = _files(tmp_path, _base(), _tuned())

    _, report = extract(base, tuned, [2, 1, 2])

    assert report == {
        "profile": "generic",
        "mode": "fixed",
        "value": 0.0,
        "unchanged_tensors": 1,
        "changed_but_not_extracted": ["norm"],
        "per_cap": {
            1: {
                "modules": 1,
                "rank_min": 1,
                "rank_max": 1,
                "rank_mean": 1.0,
                "dropped_mean": pytest.approx((4 / 13) ** 0.5),
                "dropped_worst": pytest.approx((4 / 13) ** 0.5),
            },
            2: {
                "modules": 1,
                "rank_min": 2,
                "rank_max": 2,
                "rank_mean": 2.0,
                "dropped_mean": 0.0,
                "dropped_worst": 0.0,
            },
        },
    }


def test_each_cap_gets_its_own_state_dict_in_the_chosen_dtype_and_prefix(tmp_path: Path) -> None:
    base, tuned = _files(tmp_path, _base(), _tuned())

    states, _ = extract(base, tuned, [2, 1], out_prefix="")

    assert list(states) == [1, 2]
    assert states[1]["a.lora_A.weight"].shape == (1, 3)
    assert states[1]["a.lora_B.weight"].dtype == torch.bfloat16
    assert float(states[1]["a.alpha"]) == 1.0


def test_a_dynamic_mode_picks_each_layers_rank_under_the_cap(tmp_path: Path) -> None:
    base, tuned = _files(tmp_path, _base(), _tuned())

    states, report = extract(base, tuned, [2], mode="threshold", value=2.5)

    assert float(states[2]["diffusion_model.a.alpha"]) == 1.0
    assert (report["mode"], report["value"], report["per_cap"][2]["rank_max"]) == ("threshold", 2.5, 1)


def _split_rows(name: str, diff: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    return [(f"{name}.top", diff[:1]), (f"{name}.bottom", diff[1:])]


def _two_d_except_b(name: str, shape: Sequence[int]) -> bool:
    return len(shape) == 2 and name != "b"


def test_a_profile_splits_modules_and_skips_parts_that_did_not_change(tmp_path: Path) -> None:
    tuned = _tuned()
    tuned["a.weight"] = _base()["a.weight"] + torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    tuned["b.weight"] = torch.zeros(2, 2)
    base, tuned_path = _files(tmp_path, _base(), tuned)
    profile = ExtractProfile(name="rows", include=_two_d_except_b, split=_split_rows)

    states, report = extract(base, tuned_path, [1], profile=profile, out_prefix="")

    assert sorted(states[1]) == ["a.bottom.alpha", "a.bottom.lora_A.weight", "a.bottom.lora_B.weight"]
    assert report["profile"] == "rows"
    assert report["changed_but_not_extracted"] == ["b", "norm"]


def test_progress_wraps_the_weight_names(tmp_path: Path) -> None:
    base, tuned = _files(tmp_path, _base(), _tuned())
    seen: List[List[str]] = []

    def progress(names: List[str]) -> List[str]:
        seen.append(list(names))
        return names

    extract(base, tuned, [1], progress=progress)

    assert seen == [["a.weight", "b.weight", "norm.weight"]]


def test_the_generic_profile_claims_no_checkpoint() -> None:
    assert GENERIC.detect(["a.weight"]) is False


def test_an_unknown_mode_fails_before_reading_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        extract(tmp_path / "missing", tmp_path / "missing", [1], mode="quantile")


def test_checkpoints_with_different_tensors_are_an_error(tmp_path: Path) -> None:
    tuned = _tuned()
    tuned["extra.weight"] = torch.ones(2, 2)
    base, tuned_path = _files(tmp_path, _base(), tuned)

    with pytest.raises(ValueError, match=r"different tensors, e.g. \['extra.weight'\] \(1 differ\)"):
        extract(base, tuned_path, [1])


@pytest.mark.parametrize(
    ("dtype", "stored"),
    [(torch.int8, "I8"), (torch.uint8, "U8"), (torch.float8_e4m3fn, "F8_E4M3"), (torch.float8_e5m2, "F8_E5M2")],
)
def test_a_quantized_checkpoint_is_refused(tmp_path: Path, dtype: torch.dtype, stored: str) -> None:
    base = {"q.weight": torch.ones(2, 2).to(dtype)}
    base_path, tuned_path = _files(tmp_path, base, base)

    with pytest.raises(ValueError, match=f"q.weight is stored as {stored}"):
        extract(base_path, tuned_path, [1])


def test_checkpoints_with_no_changed_target_are_an_error(tmp_path: Path) -> None:
    tuned = _base()
    tuned["norm.weight"] = torch.zeros(3)
    base, tuned_path = _files(tmp_path, _base(), tuned)

    with pytest.raises(ValueError, match="no LoRA target changed"):
        extract(base, tuned_path, [1])
