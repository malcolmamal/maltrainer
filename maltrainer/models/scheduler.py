"""
Noise schedulers for training and inference.
"""

from typing import Any, cast

from diffusers import FlowMatchEulerDiscreteScheduler


def get_flow_match_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Get the flow matching scheduler for Z-Image."""
    scheduler: FlowMatchEulerDiscreteScheduler = cast(Any, FlowMatchEulerDiscreteScheduler)(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    return scheduler
