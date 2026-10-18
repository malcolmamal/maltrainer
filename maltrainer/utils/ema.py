"""
Exponential moving average of trainable weights: adapters on the GPU, whole models in CPU RAM.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

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


class OffloadedParameterEMA:
    """fp32 shadow of a whole model in CPU RAM, for full finetuning where a GPU copy will not fit.

    Each update copies the live weights into a pinned staging buffer (one GPU -> CPU transfer),
    then averages on a background thread, so the CPU work overlaps the next training steps. The
    next update, a save or a state dump waits for it first. ``every`` > 1 updates less often with
    the decay raised to that power, which keeps the same time constant in optimizer steps.
    """

    def __init__(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]], decay: float = 0.999,
                 warmup: bool = True, every: int = 1):
        named = [(name, p) for name, p in named_parameters if p.requires_grad]
        self.names: List[str] = [name for name, _ in named]
        self.parameters: List[torch.nn.Parameter] = [p for _, p in named]
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.every = max(int(every), 1)
        self.updates = 0
        self.calls = 0
        pin = torch.cuda.is_available()
        self.shadow = [p.detach().to("cpu", torch.float32) for p in self.parameters]
        self.staging = [torch.empty(p.shape, dtype=p.dtype, pin_memory=pin) for p in self.parameters]
        self._thread: Optional[threading.Thread] = None

    def current_decay(self) -> float:
        decay = self.decay if not self.warmup else min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        return decay ** self.every

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @torch.no_grad()
    def update(self) -> None:
        self.calls += 1
        if self.calls % self.every:
            return
        self.wait()
        for staged, param in zip(self.staging, self.parameters):
            staged.copy_(param.detach(), non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()  # staged before the next step moves the weights
        weight = 1.0 - self.current_decay()
        self.updates += 1
        self._thread = threading.Thread(target=self._average, args=(weight,), daemon=True)
        self._thread.start()

    def _average(self, weight: float, chunk: int = 1 << 20) -> None:
        # Chunked so the fp32 copy of the staged bf16 stays in cache: 1.1 s for 5.08B on a
        # 9800X3D, against 2.9 s converting whole tensors (it is memory-bound either way).
        with torch.no_grad():
            for shadow, staged in zip(self.shadow, self.staging):
                flat_shadow, flat_staged = shadow.view(-1), staged.view(-1)
                for i in range(0, flat_shadow.numel(), chunk):
                    flat_shadow[i:i + chunk].lerp_(flat_staged[i:i + chunk].float(), weight)

    def averaged(self) -> Dict[str, torch.Tensor]:
        """name -> fp32 CPU average (the live shadow tensors; do not modify)."""
        self.wait()
        return dict(zip(self.names, self.shadow))

    @contextmanager
    def applied(self) -> Iterator[None]:
        """The average in the live weights for the duration (a preview render), then the raw
        weights back, bit for bit. The raw weights wait in the pinned staging buffers, which are
        free between updates, so this needs no extra memory; it moves the model over PCIe twice."""
        self.wait()
        with torch.no_grad():
            for staged, param, shadow in zip(self.staging, self.parameters, self.shadow):
                staged.copy_(param.detach(), non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            for param, shadow in zip(self.parameters, self.shadow):
                param.copy_(shadow.to(param.device, non_blocking=True))
        try:
            yield
        finally:
            with torch.no_grad():
                for staged, param in zip(self.staging, self.parameters):
                    param.copy_(staged.to(param.device, non_blocking=True))
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()

    def state(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        return self.averaged(), {"updates": self.updates, "calls": self.calls}

    def load_state(self, tensors: Dict[str, torch.Tensor], counters: Dict[str, int]) -> None:
        self.wait()
        missing = [name for name in self.names if name not in tensors]
        if missing:
            raise ValueError(f"EMA state lacks {len(missing)} trained parameters, e.g. {missing[:3]}")
        for name, shadow in zip(self.names, self.shadow):
            shadow.copy_(tensors[name])
        self.updates = int(counters.get("updates", 0))
        self.calls = int(counters.get("calls", 0))
