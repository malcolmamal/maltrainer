from __future__ import annotations

import pytest
import torch

from maltrainer.finetune.extract import factor, keep_count, strip_prefix


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["model.diffusion_model.a.weight", "model.diffusion_model.b.weight"], "model.diffusion_model."),
        (["diffusion_model.a.weight", "diffusion_model.b.weight"], "diffusion_model."),
        (["diffusion_model.a.weight", "b.weight"], ""),
        (["a.weight"], ""),
        ([], ""),
    ],
)
def test_the_container_prefix_is_the_one_every_key_carries(keys: list[str], expected: str) -> None:
    assert strip_prefix(keys) == expected


S = torch.tensor([4.0, 2.0, 1.0])


@pytest.mark.parametrize(
    ("mode", "value", "cap", "expected"),
    [
        ("fixed", 0.0, 2, 2),
        ("fixed", 0.0, 9, 3),
        ("fixed", 0.0, 0, 1),
        ("fro", 0.5, 3, 1),
        ("fro", 0.9, 3, 2),
        ("fro", 0.99, 3, 3),
        ("fro", 0.99, 2, 2),
        ("ratio", 0.5, 3, 2),
        ("ratio", 0.25, 3, 3),
        ("ratio", 0.25, 1, 1),
        ("threshold", 1.5, 3, 2),
        ("threshold", 0.0, 2, 2),
        ("threshold", 10.0, 3, 1),
    ],
)
def test_keep_count(mode: str, value: float, cap: int, expected: int) -> None:
    assert keep_count(S, mode, value, cap) == expected


@pytest.mark.parametrize("mode", ["fro", "ratio", "threshold"])
def test_a_layer_with_no_singular_energy_keeps_one(mode: str) -> None:
    assert keep_count(torch.zeros(3), mode, 0.5, 3) == 1


def test_an_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        keep_count(S, "quantile", 0.5, 3)


DIFF = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def test_each_cap_gets_its_best_low_rank_factors_from_one_svd() -> None:
    result = factor(DIFF, [1, 2], "fixed", 0.0, "cpu")

    down, up, k, error = result[1]
    assert (down.shape, up.shape, k) == ((1, 3), (2, 1), 1)
    torch.testing.assert_close(up @ down, torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert error == pytest.approx((4 / 13) ** 0.5)
    down, up, k, error = result[2]
    assert (down.shape, up.shape, k, error) == ((2, 3), (2, 2), 2, 0.0)
    torch.testing.assert_close(up @ down, DIFF)


def test_up_and_down_share_the_singular_values_evenly() -> None:
    down, up, _, _ = factor(DIFF, [2], "fixed", 0.0, "cpu")[2]

    torch.testing.assert_close(torch.linalg.vector_norm(up, dim=0), torch.tensor([3.0, 2.0]).sqrt())
    torch.testing.assert_close(torch.linalg.vector_norm(down, dim=1), torch.tensor([3.0, 2.0]).sqrt())


def test_factor_follows_the_rank_mode() -> None:
    _, _, k, error = factor(DIFF, [2], "threshold", 2.5, "cpu")[2]

    assert k == 1
    assert error == pytest.approx((4 / 13) ** 0.5)


def test_a_zero_difference_reports_nothing_dropped() -> None:
    down, up, k, error = factor(torch.zeros(2, 3), [1], "fixed", 0.0, "cpu")[1]

    assert (k, error) == (1, 0.0)
    torch.testing.assert_close(up @ down, torch.zeros(2, 3))


def test_factors_come_back_in_fp32_from_any_input_dtype() -> None:
    down, up, _, _ = factor(DIFF.to(torch.bfloat16), [2], "fixed", 0.0, "cpu")[2]

    assert (down.dtype, up.dtype) == (torch.float32, torch.float32)
    torch.testing.assert_close(up @ down, DIFF)
