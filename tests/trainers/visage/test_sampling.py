"""The preview sampler: Visage-Image's shifted SDE-churn Euler with classifier-free guidance against a zero caption."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from maltrainer.trainers.visage.sampling import churned, sample_latents, shifted_timesteps

CPU = torch.device("cpu")
CHANNELS = 8


class Recorder:
    """A DiT with a constant velocity that records what it was called with."""

    def __init__(self, velocity: float = 0.0) -> None:
        self.config = SimpleNamespace(in_channels=CHANNELS)
        self.velocity = velocity
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        self.calls.append((x.clone(), t.clone(), text.clone()))
        return torch.full_like(x, self.velocity)


def _noise(seed: int, grid: tuple[int, int] = (2, 3)) -> torch.Tensor:
    return torch.randn((1, CHANNELS, *grid), generator=torch.Generator().manual_seed(seed))


def test_the_times_run_from_one_to_zero_warped_toward_the_noisy_end() -> None:
    times = shifted_timesteps(4, 3.3)

    assert times.shape == (5,) and times[0].item() == pytest.approx(1.0) and times[-1].item() == 0.0
    assert (times[1:] < times[:-1]).all()
    uniform = torch.linspace(1.0, 0.0, 5)
    assert times.tolist() == pytest.approx((3.3 * uniform / (1 + 2.3 * uniform)).tolist())
    assert times[2] > 0.5  # the middle step is still mostly noise


def test_a_shift_of_one_leaves_the_grid_uniform() -> None:
    assert shifted_timesteps(4, 1.0).tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_churn_renoises_to_the_new_time_with_exactly_the_missing_variance() -> None:
    x, noise = torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)

    out, t_hat = churned(x, 0.5, 0.3, noise)

    scale = (1 - t_hat) / (1 - 0.5)
    assert t_hat == pytest.approx(0.65)
    # x_t = (1-t) x0 + t n; the result must be (1 - t_hat) x0 + t_hat n': signal scaled, noise variance t_hat^2
    assert torch.allclose(out, scale * x + (t_hat**2 - (scale * 0.5) ** 2) ** 0.5 * noise)


def test_churn_stops_at_the_ceiling_of_0_9() -> None:
    _, t_hat = churned(torch.zeros(1), 0.8, 0.5, torch.zeros(1))

    assert t_hat == 0.9


def test_without_churn_the_sample_is_unchanged_in_value() -> None:
    x = torch.randn(1, 4, 2, 2)

    out, t_hat = churned(x, 0.5, 0.0, torch.randn_like(x))

    assert t_hat == 0.5 and torch.equal(out, x)


def test_euler_with_a_constant_velocity_moves_the_noise_by_that_velocity_over_unit_time() -> None:
    model = Recorder(velocity=0.5)
    cond = torch.ones(1, 3, 4)

    out = sample_latents(model, cond, (2, 3), 7, CPU, torch.float32, steps=6, guidance=1.0, solver="euler")

    assert torch.allclose(out, _noise(7) - 0.5, atol=1e-5)
    assert len(model.calls) == 6


def test_the_start_is_the_seeded_noise_and_the_first_call_is_at_t_one() -> None:
    model = Recorder()

    out = sample_latents(model, torch.ones(1, 3, 4), (2, 3), 3, CPU, torch.float32, steps=4, guidance=1.0)

    assert torch.equal(model.calls[0][0], _noise(3)) and model.calls[0][1].item() == 1.0
    assert out.shape == (1, CHANNELS, 2, 3)


def test_guidance_runs_the_conditional_and_a_zero_caption_in_one_batch_and_mixes_their_velocities() -> None:
    class Marker(Recorder):
        def __call__(self, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            super().__call__(x, t, text)
            return text.mean(dim=(1, 2)).view(-1, 1, 1, 1).expand_as(x)  # 1 for the caption, 0 for the zeros

    model = Marker()
    cond = torch.ones(1, 3, 4)

    out = sample_latents(model, cond, (2, 3), 7, CPU, torch.float32, steps=5, guidance=3.0, solver="euler")

    x, _, text = model.calls[0]
    assert x.shape[0] == 2 and torch.equal(text[0], cond[0]) and torch.equal(text[1], torch.zeros(3, 4))
    assert torch.allclose(out, _noise(7) - 3.0, atol=1e-5)  # uncond + 3 (cond - uncond) = 3


def test_a_guidance_of_one_runs_the_conditional_branch_alone() -> None:
    model = Recorder()

    sample_latents(model, torch.ones(1, 3, 4), (2, 3), 7, CPU, torch.float32, steps=3, guidance=1.0)

    assert model.calls[0][0].shape[0] == 1 and torch.equal(model.calls[0][2], torch.ones(1, 3, 4))


def test_the_sde_solver_renoises_between_steps_and_the_euler_solver_does_not() -> None:
    sde, euler = Recorder(), Recorder()
    cond = torch.ones(1, 3, 4)

    sample_latents(sde, cond, (2, 3), 7, CPU, torch.float32, steps=8, guidance=1.0, solver="sde", churn=0.3)
    sample_latents(euler, cond, (2, 3), 7, CPU, torch.float32, steps=8, guidance=1.0, solver="euler", churn=0.3)

    times = shifted_timesteps(8, 3.3).tolist()[:-1]
    assert [c[1].item() for c in euler.calls] == pytest.approx(times)
    assert sde.calls[0][1].item() == 1.0  # t = 1 is above the ceiling: no churn
    expected = [min(t * 1.3, 0.9) if t < 0.9 else t for t in times[1:]]
    assert [c[1].item() for c in sde.calls[1:]] == pytest.approx(expected)
    assert torch.equal(sde.calls[1][0], euler.calls[1][0])  # still above the ceiling: nothing re-noised yet
    assert not torch.equal(sde.calls[-1][0], euler.calls[-1][0])  # re-noised once t is under it


def test_zero_churn_makes_the_sde_solver_the_euler_one() -> None:
    cond = torch.ones(1, 3, 4)

    sde = sample_latents(Recorder(0.3), cond, (2, 3), 7, CPU, torch.float32, steps=5, solver="sde", churn=0.0)
    euler = sample_latents(Recorder(0.3), cond, (2, 3), 7, CPU, torch.float32, steps=5, solver="euler")

    assert torch.equal(sde, euler)


def test_the_same_seed_gives_the_same_latents_and_a_new_one_gives_new_noise() -> None:
    cond = torch.ones(1, 3, 4)

    def run(seed: int) -> torch.Tensor:
        return sample_latents(Recorder(0.2), cond, (2, 3), seed, CPU, torch.float32, steps=6)

    assert torch.equal(run(1), run(1)) and not torch.equal(run(1), run(2))


def test_progress_is_reported_after_every_step() -> None:
    seen: list[tuple[int, int]] = []

    sample_latents(
        Recorder(),
        torch.ones(1, 3, 4),
        (2, 3),
        0,
        CPU,
        torch.float32,
        steps=4,
        on_step=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_the_model_sees_the_requested_dtype() -> None:
    model = Recorder()

    sample_latents(model, torch.ones(1, 3, 4), (2, 3), 0, CPU, torch.bfloat16, steps=2)

    assert model.calls[0][0].dtype == torch.bfloat16 and model.calls[0][2].dtype == torch.bfloat16


def test_an_unknown_solver_is_refused() -> None:
    with pytest.raises(ValueError, match=r"sample\.solver must be one of \('sde', 'euler'\), got 'heun'"):
        sample_latents(Recorder(), torch.ones(1, 3, 4), (2, 3), 0, CPU, torch.float32, solver="heun")
