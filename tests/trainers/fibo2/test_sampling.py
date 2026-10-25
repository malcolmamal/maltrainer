"""The preview sampler: BRIA's shifted flow-matching Euler with classifier-free guidance."""

from __future__ import annotations

import math

import pytest
import torch

from maltrainer.trainers.fibo2.sampling import sample_latents, sampling_sigmas

CPU = torch.device("cpu")


def test_the_sigmas_run_from_one_to_zero_decreasing() -> None:
    sigmas = sampling_sigmas(10, 1024)

    assert sigmas.shape == (11,) and sigmas[0].item() == pytest.approx(1.0) and sigmas[-1].item() == 0.0
    assert (sigmas[1:] < sigmas[:-1]).all()


def test_the_shift_follows_the_flux2_formula_and_grows_with_the_image_size() -> None:
    steps, tokens = 4, 4096
    mu = 1.15  # the maximum, at 4096 tokens
    expected = [math.exp(mu) / (math.exp(mu) + (1 / s - 1)) for s in (1.0, 0.75, 0.5, 0.25)]

    assert sampling_sigmas(steps, tokens)[:-1].tolist() == pytest.approx(expected, abs=1e-6)
    assert (sampling_sigmas(10, 4096)[1:-1] > sampling_sigmas(10, 256)[1:-1]).all()


class Oracle:
    """A DiT that knows the clean latent: its velocity (x - x0) / sigma is exact along the straight path."""

    def __init__(self, clean: torch.Tensor):
        self.clean = clean
        self.calls = 0

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (x - self.clean) / sigma.view(1, 1, 1, 1)


def test_euler_with_the_exact_velocity_lands_on_the_clean_latent() -> None:
    clean = torch.randn(1, 128, 2, 3)
    model = Oracle(clean)
    context = torch.zeros(1, 2, 4)

    out = sample_latents(model, context, context, (2, 3), 6, 1.0, 5, CPU, torch.float32)

    assert torch.allclose(out, clean, atol=1e-4) and model.calls == 6


class Marker:
    """Velocity 1 for the conditional features (ones), 0 for the unconditional ones (zeros)."""

    def __call__(self, x: torch.Tensor, _sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, context.mean().item())


@pytest.mark.parametrize(("guidance", "velocity"), [(1.0, 1.0), (3.0, 3.0), (0.0, 0.0)])
def test_the_guided_velocity_is_uncond_plus_guidance_times_the_difference(guidance: float, velocity: float) -> None:
    ones, zeros = torch.ones(1, 2, 4), torch.zeros(1, 2, 4)
    noise = torch.randn((1, 128, 2, 2), generator=torch.Generator().manual_seed(9))

    out = sample_latents(Marker(), ones, zeros, (2, 2), 5, guidance, 9, CPU, torch.float32)

    # the sigmas fall by one in total, so the latent moves by -velocity from the noise
    assert torch.allclose(out, noise - velocity, atol=1e-5)


def test_without_guidance_the_unconditional_features_are_never_run() -> None:
    model = Oracle(torch.zeros(1, 128, 2, 2))

    sample_latents(model, torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), (2, 2), 4, 1.0, 1, CPU, torch.float32)
    once = model.calls
    sample_latents(model, torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), (2, 2), 4, 5.0, 1, CPU, torch.float32)

    assert once == 4 and model.calls == 4 + 8


def test_the_seed_decides_the_noise_and_the_training_rng_is_left_alone() -> None:
    ones, zeros = torch.ones(1, 2, 4), torch.zeros(1, 2, 4)
    torch.manual_seed(0)
    state = torch.get_rng_state()

    first = sample_latents(Marker(), ones, zeros, (2, 2), 3, 5.0, 7, CPU, torch.float32)
    again = sample_latents(Marker(), ones, zeros, (2, 2), 3, 5.0, 7, CPU, torch.float32)
    other = sample_latents(Marker(), ones, zeros, (2, 2), 3, 5.0, 8, CPU, torch.float32)

    assert torch.equal(first, again) and not torch.equal(first, other)
    assert torch.equal(torch.get_rng_state(), state)


def test_the_model_gets_the_compute_dtype_and_the_latents_stay_fp32() -> None:
    seen: list[torch.dtype] = []

    def model(x: torch.Tensor, _sigma: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        seen.append(x.dtype)
        return torch.zeros_like(x)

    out = sample_latents(model, torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), (2, 2), 2, 1.0, 1, CPU, torch.bfloat16)

    assert set(seen) == {torch.bfloat16} and out.dtype == torch.float32


def test_the_step_callback_reports_every_step_in_order() -> None:
    seen: list[tuple[int, int]] = []
    zeros = torch.zeros(1, 2, 4)

    sample_latents(
        Marker(), zeros, zeros, (2, 2), 4, 1.0, 1, CPU, torch.float32, lambda done, total: seen.append((done, total))
    )

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
