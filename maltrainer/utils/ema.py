"""
Exponential moving average of trainable adapter weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, List

import torch


class ParameterEMA:
    """fp32 shadow copy of a set of parameters, swapped in for saving.

    The decay ramps in as ``min(decay, (1 + n) / (10 + n))``, the same warm-up Fizgig uses
    with its 0.98 default, so the average tracks the fast early movement of a fresh LoRA
    instead of dragging its zero-initialized start along for hundreds of steps.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.98, warmup: bool = True):
        self.parameters: List[torch.nn.Parameter] = [p for p in parameters if p.requires_grad]
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0
        self.shadow = [p.detach().float().clone() for p in self.parameters]

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self) -> None:
        weight = 1.0 - self.current_decay()
        for shadow, param in zip(self.shadow, self.parameters):
            shadow.lerp_(param.detach().float(), weight)
        self.updates += 1

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Temporarily load the averaged weights into the live parameters."""
        with torch.no_grad():
            backup = [param.detach().clone() for param in self.parameters]
            for shadow, param in zip(self.shadow, self.parameters):
                param.copy_(shadow.to(param.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for saved, param in zip(backup, self.parameters):
                    param.copy_(saved)
