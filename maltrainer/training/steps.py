"""How many optimizer steps a run takes, and the order an epoch visits the samples in."""

from __future__ import annotations

import math
import random
from typing import Any


def plan_training_steps(num_samples: int, train_config: dict[str, Any]) -> tuple[int, int, int]:
    """(steps_per_epoch, num_epochs, total_steps); max_steps wins. An epoch yields num_samples / grad_accum steps."""
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_samples / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps * grad_accum / max(num_samples, 1)), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, max(num_epochs * max(num_samples, 1) // grad_accum, 1)


def epoch_order(num_samples: int, repeats: Any, num_processes: int, process_index: int) -> list[int]:
    """Every sample ``repeats`` times, shuffled (Python's ``random``); with several processes, this one's share."""
    copies = max(int(repeats or 1), 1)
    order = [i for i in range(num_samples) for _ in range(copies)]
    random.shuffle(order)
    return order[process_index::num_processes] if num_processes > 1 else order
