"""Krea 2's preview sampler: the shifted schedule, the seeded noise, Euler steps and CFG."""

from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.krea2.sampling import initial_noise, sample, schedule


def test_the_schedule_runs_from_one_to_zero_in_steps_plus_one_points() -> None:
    sigmas = schedule(4, 256)
    assert sigmas.dtype == torch.float32 and sigmas.shape == (5,)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())


@pytest.mark.parametrize("tokens, mu", [(256, 0.5), (6400, 1.15), (3328, 0.825)])
def test_the_schedule_is_fizgigs_time_shift_by_the_image_tokens(tokens: int, mu: float) -> None:
    # Fizgig / ai-toolkit: e^mu / (e^mu + (1 / t - 1)) on a uniform grid, mu 0.5 at 256 tokens to 1.15 at 6400
    t = torch.linspace(1, 0, 5, dtype=torch.float64)[1:-1]
    expected = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
    torch.testing.assert_close(schedule(4, tokens)[1:-1], expected.float())


def test_more_tokens_keep_sigma_higher_longer() -> None:
    assert bool((schedule(8, 4096)[1:-1] > schedule(8, 1024)[1:-1]).all())


def test_a_schedule_needs_a_step() -> None:
    with pytest.raises(ValueError, match="steps must be at least 1, got 0"):
        schedule(0, 256)


def test_the_noise_is_the_seeds_fp32_cpu_draw_at_an_eighth_of_the_size() -> None:
    noise = initial_noise(7, 32, 48, channels=4)
    expected = torch.randn((1, 4, 4, 6), generator=torch.Generator().manual_seed(7))
    assert noise.dtype == torch.float32 and torch.equal(noise, expected)
    assert initial_noise(7, 16, 16).shape == (1, 16, 2, 2)


def test_the_size_message_names_the_nearest_sizes_that_work() -> None:
    from maltrainer.models.krea2.sampling import size_problem

    assert size_problem(1072, 1472) is None
    assert size_problem(1080, 1480) == (
        "sample.width / height must be multiples of 16, got 1080x1480: use 1072 or 1088 for 1080, 1472 or 1488 for 1480"
    )
    assert (
        size_problem(8, 0)
        == "sample.width / height must be multiples of 16, got 8x0: use at least 16 for 8, at least 16 for 0"
    )


@pytest.mark.parametrize("height, width", [(24, 32), (32, 40), (0, 16)])
def test_sizes_off_the_16_px_grid_are_refused(height: int, width: int) -> None:
    with pytest.raises(ValueError, match=r"sample\.width / height must be multiples of 16"):
        initial_noise(0, height, width)


class ConstantVelocity:
    """A DiT whose velocity is fixed per prompt (the prompt's first value), recording each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float, int]] = []

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self.calls.append((float(sigma), float(context.flatten()[0]), context.dim()))
        return torch.full_like(x, float(context.flatten()[0]), dtype=torch.bfloat16)


def test_euler_without_guidance_moves_the_noise_by_the_velocity_over_the_whole_schedule() -> None:
    dit = ConstantVelocity()
    positive = torch.full((3, 2, 4), 0.5)
    steps: list[tuple[int, int]] = []

    out = sample(
        dit,
        positive,
        16,
        32,
        steps=3,
        seed=1,
        cfg=1.0,
        negative=torch.zeros(3, 2, 4),
        channels=4,
        on_step=lambda done, total: steps.append((done, total)),
    )

    torch.testing.assert_close(out, initial_noise(1, 16, 32, 4) - 0.5)  # sigma 1 -> 0, x += (0 - 1) * v
    assert [c[1] for c in dit.calls] == [0.5] * 3  # cfg 1: the negative is never run
    assert [round(c[0], 6) for c in dit.calls] == [
        round(float(s), 6) for s in schedule(3, 2)[:-1]
    ]  # a 2 x 4 latent: 1 x 2 tokens
    assert all(c[2] == 4 for c in dit.calls)  # the text gets its batch axis
    assert steps == [(1, 3), (2, 3), (3, 3)]


def test_cfg_is_the_negative_plus_cfg_times_the_difference() -> None:
    dit = ConstantVelocity()
    out = sample(
        dit,
        torch.full((1, 2, 2, 4), 1.0),
        16,
        16,
        steps=2,
        seed=3,
        cfg=4.0,
        negative=torch.full((2, 2, 4), 0.25),
        channels=4,
    )

    torch.testing.assert_close(out, initial_noise(3, 16, 16, 4) - (0.25 + 4.0 * (1.0 - 0.25)))
    assert [c[1] for c in dit.calls] == [1.0, 0.25, 1.0, 0.25]


def test_without_a_negative_there_is_no_guidance() -> None:
    dit = ConstantVelocity()
    sample(dit, torch.full((2, 2, 4), 1.0), 16, 16, steps=2, cfg=4.5, channels=4)
    assert len(dit.calls) == 2
