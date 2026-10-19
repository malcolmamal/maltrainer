from __future__ import annotations

from typing import List

import pytest
import torch

from maltrainer.models.moss_image.sampling import UNCOND_SKIP, initial_noise, sample, schedule

POS = torch.zeros(3, 4)
NEG = torch.ones(3, 4)


class Recorder:
    """A stand-in DiT: ``velocity(x, context)`` is what it predicts; every call is logged."""

    def __init__(self, velocity=None) -> None:
        self.velocity = velocity or (lambda x, _context: torch.full_like(x, 0.5))
        self.calls: List[dict] = []

    def __call__(self, x, sigma, context, skip_layers=None):
        self.calls.append({"sigma": sigma.item(), "context": context, "skip": skip_layers})
        return self.velocity(x, context)


def _by_context(x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, 3.0 if context.sum() == 0 else 1.0)


@pytest.mark.parametrize(
    ("steps", "shift", "expected"),
    [
        (1, 5.0, [1.0, 0.0]),
        (2, 5.0, [1.0, 0.024, 0.0]),
        (5, 5.0, [1.0, 0.938, 0.835, 0.63, 0.024, 0.0]),
        (4, 1.0, [1.0, 0.667, 0.334, 0.001, 0.0]),
    ],
)
def test_the_schedule_is_the_shifted_curve_snapped_to_the_thousand_training_sigmas(
    steps: int, shift: float, expected: List[float]
) -> None:
    out = schedule(steps, shift)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor(expected), atol=1e-6, rtol=0)


def test_a_dense_schedule_drops_the_repeats_snapping_creates_near_sigma_one() -> None:
    out = schedule(1000)

    assert 1 < len(out) < 1001
    assert (out[1:] < out[:-1]).all()
    assert out[-1] == 0.0
    torch.testing.assert_close(out[:-1] * 1000, (out[:-1] * 1000).round(), atol=1e-3, rtol=0)


@pytest.mark.parametrize(("steps", "shift"), [(0, 5.0), (-3, 5.0), (4, 0.0), (4, -1.0)])
def test_a_schedule_needs_positive_steps_and_shift(steps: int, shift: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        schedule(steps, shift)


def test_initial_noise_is_fp32_gaussian_on_the_downsampled_grid_and_fixed_by_the_seed() -> None:
    noise = initial_noise(5, 32, 48)

    expected = torch.randn((1, 128, 2, 3), generator=torch.Generator("cpu").manual_seed(5))
    assert noise.dtype == torch.float32
    torch.testing.assert_close(noise, expected)
    assert not torch.allclose(initial_noise(6, 32, 48), noise)


@pytest.mark.parametrize(("height", "width"), [(30, 32), (32, 40), (17, 16)])
def test_initial_noise_needs_sizes_that_are_multiples_of_16(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        initial_noise(0, height, width)


def test_a_constant_velocity_moves_the_noise_by_minus_that_velocity_over_the_whole_schedule() -> None:
    dit = Recorder()

    out = sample(dit, POS, 32, 32, steps=5, seed=3, cfg=1.0)

    torch.testing.assert_close(out, initial_noise(3, 32, 32) - 0.5)
    assert [call["sigma"] for call in dit.calls] == pytest.approx([1.0, 0.938, 0.835, 0.63, 0.024])
    assert all(call["context"].shape == (1, 3, 4) and call["skip"] is None for call in dit.calls)


def test_euler_steps_scale_the_latent_by_one_plus_the_sigma_decrement() -> None:
    dit = Recorder(lambda x, _context: x)

    out = sample(dit, POS, 16, 16, steps=2, seed=1, cfg=1.0)

    torch.testing.assert_close(out, initial_noise(1, 16, 16) * (0.024 * 0.976), atol=1e-6, rtol=1e-5)


def test_cfg_extrapolates_from_the_negative_branch_which_skips_block_22() -> None:
    dit = Recorder(_by_context)

    out = sample(dit, POS, 16, 16, steps=2, seed=2, cfg=3.0, negative=NEG)

    torch.testing.assert_close(out, initial_noise(2, 16, 16) - (1.0 + 3.0 * (3.0 - 1.0)))
    assert [call["skip"] for call in dit.calls] == [None, UNCOND_SKIP] * 2
    assert UNCOND_SKIP == (22,)
    assert [call["context"].sum().item() for call in dit.calls[:2]] == [0.0, 12.0]


def test_the_negative_branch_skips_whatever_layers_it_is_told_to() -> None:
    dit = Recorder()

    sample(dit, POS, 16, 16, steps=1, cfg=2.0, negative=NEG, uncond_skip=[4, 7])

    assert dit.calls[1]["skip"] == (4, 7)


@pytest.mark.parametrize(("cfg", "negative"), [(1.0, NEG), (3.0, None)])
def test_one_prediction_per_step_when_there_is_nothing_to_guide_against(cfg: float, negative) -> None:
    dit = Recorder()

    sample(dit, POS, 16, 16, steps=3, cfg=cfg, negative=negative)

    assert len(dit.calls) == len(schedule(3)) - 1


def test_a_batched_context_is_passed_through_and_progress_is_reported_after_every_step() -> None:
    dit = Recorder()
    progress: List[tuple] = []

    sample(dit, POS[None], 16, 16, steps=5, cfg=1.0, on_step=lambda done, total: progress.append((done, total)))

    assert dit.calls[0]["context"].shape == (1, 3, 4)
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_predictions_are_combined_in_fp32_and_the_result_carries_no_graph() -> None:
    weight = torch.nn.Parameter(torch.ones(()))
    dit = Recorder(lambda x, _context: (x * weight).to(torch.bfloat16))

    out = sample(dit, POS, 16, 16, steps=2, cfg=1.0)

    assert out.dtype == torch.float32
    assert not out.requires_grad


def test_the_same_seed_renders_the_same_latents() -> None:
    first = sample(Recorder(_by_context), POS, 16, 32, steps=3, seed=9, cfg=2.0, negative=NEG)
    second = sample(Recorder(_by_context), POS, 16, 32, steps=3, seed=9, cfg=2.0, negative=NEG)

    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.shape == (1, 128, 1, 2)
