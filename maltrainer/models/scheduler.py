"""
Noise schedulers for training and inference.
"""

from diffusers import FlowMatchEulerDiscreteScheduler


def get_flow_match_scheduler():
    """Get the flow matching scheduler for Z-Image."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
