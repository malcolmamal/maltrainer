from __future__ import annotations

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler

from maltrainer.models.scheduler import get_flow_match_scheduler


def _shifted(sigma: float, shift: float = 3.0) -> float:
    return shift * sigma / (1 + (shift - 1) * sigma)


def test_the_scheduler_is_a_static_shift_flow_match_schedule_over_1000_steps() -> None:
    scheduler = get_flow_match_scheduler()

    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler.config.num_train_timesteps == 1000
    assert scheduler.config.shift == 3.0
    assert scheduler.config.use_dynamic_shifting is False


@pytest.mark.parametrize("index", [0, 499, 750, 999])
def test_training_sigmas_are_shifted_towards_noise_by_three(index: int) -> None:
    sigma = (1000 - index) / 1000

    scheduler = get_flow_match_scheduler()

    assert scheduler.sigmas[index].item() == pytest.approx(_shifted(sigma), rel=1e-5)
    assert scheduler.timesteps[index].item() == pytest.approx(_shifted(sigma) * 1000, rel=1e-5)


def test_each_call_returns_an_independent_scheduler() -> None:
    first = get_flow_match_scheduler()
    first.set_timesteps(2)

    assert len(get_flow_match_scheduler().timesteps) == 1000
