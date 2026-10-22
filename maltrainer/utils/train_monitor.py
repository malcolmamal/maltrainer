"""
Training-health summaries: one log line every ``every`` steps that says whether a run is learning
and catches a run that breaks.

A single step's loss says little: it is one image at one noise level, and on MOSS-Image it swings
between ~0.25 and ~0.9 with the sigma drawn. So the summary averages over the window, and splits
the loss into noise bands. Learning a likeness shows first in the low-noise band, which the
timestep sampler visits least.

    [monitor] step 400/8932 | loss 0.5821 (low 0.214 n=13, mid 0.512 n=41, high 0.731 n=46) |
    prior 0.00041 | signal attn +0.12 mlp +0.09 | drift attn 1.9e-03 mlp 2.4e-03 (t^0.92) |
    nonfinite 0 | 1.31 s/step | peak 21.4 GB

- **loss bands**: compare each band with itself over time, not with the others. With the
  velocity target the low-noise band has the highest loss (the noise must be read off an almost
  clean image)
- **signal** (Automagic3 only): the share of weights whose updates keep one direction for 8
  steps minus the share that flip every step. Above 0 = learning. Near 0 together with drift
  growing like t^0.5 = the weights wander (batch-1 noise; EMA averages it out); clearly
  negative = overshooting
- **drift**: |W - W0| / |W0| of a few tracked tensors, W0 = the weights when this process
  started. (t^k) is how it grew since the previous summary against the steps since the start:
  drift ~ steps^k. k near 1 = updates keep a direction; k near 0.5 = a random walk. A full
  finetune at 1e-5, batch 1, no EMA read k 0.41-0.49 and its previews wandered. Flat drift =
  nothing trains
- **nonfinite**: gradient values that were NaN / inf. The optimizer zeroes them, so they would
  otherwise pass unseen
- a loss that is NaN / inf ``stop_after`` steps in a row stops the run without saving, so the last
  good checkpoint and resume state are not overwritten
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Noise bands of the summary's loss split (sigma, the share of noise in the input)
BANDS = (("low", 0.0, 0.5), ("mid", 0.5, 0.85), ("high", 0.85, 1.01))


class NonFiniteLossError(RuntimeError):
    """The loss stayed NaN / inf; training stops without saving over the last good files."""


class WeightDrift:
    """|W - W0| / |W0| of chosen tensors, grouped by label. W0 is kept on the tensors' own
    device (pick a few tensors: their size is held twice)."""

    def __init__(self, tracked: Sequence[tuple[str, torch.nn.Parameter]]):
        self.tracked = [(label, p, p.detach().clone(), p.detach().float().norm().item()) for label, p in tracked]
        self.previous: tuple[int, float] | None = None  # (steps since the start, mean drift)

    @property
    def size_mb(self) -> float:
        return sum(w0.numel() * w0.element_size() for _, _, w0, _ in self.tracked) / 1024**2

    @torch.no_grad()
    def measure(self) -> dict[str, float]:
        sq: dict[str, float] = {}
        ref: dict[str, float] = {}
        for label, p, w0, norm in self.tracked:
            d = (p.detach().float() - w0.float()).norm().item()
            sq[label] = sq.get(label, 0.0) + d * d
            ref[label] = ref.get(label, 0.0) + norm * norm
        return {label: math.sqrt(sq[label]) / max(math.sqrt(ref[label]), 1e-30) for label in sq}

    def summary(self, steps: int) -> str:
        """``steps``: optimizer steps since W0 was taken."""
        drift = self.measure()
        text = " ".join(f"{label} {value:.1e}" for label, value in drift.items())
        mean = sum(drift.values()) / max(len(drift), 1)
        if self.previous and self.previous[1] > 0 and mean > 0 and steps > self.previous[0]:
            k = math.log(mean / self.previous[1]) / math.log(steps / self.previous[0])
            text += f" (t^{k:.2f})"
        self.previous = (steps, mean)
        return text


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


class TrainMonitor:
    def __init__(self, every: int = 100, stop_after: int = 3, drift: WeightDrift | None = None):
        self.every = max(int(every), 1)
        self.stop_after = max(int(stop_after), 1)
        self.drift = drift
        self.bad_in_a_row = 0
        self.steps_total = 0  # optimizer steps since this monitor (and the drift's W0) started
        self.last_loss: float | None = None  # the latest finite micro-step loss, kept across windows
        self._reset()

    def _reset(self) -> None:
        self.losses: list[float] = []
        self.bands: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
        self.priors: list[float] = []
        self.started = time.time()
        self.excluded = 0.0
        self.steps = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record(self, loss: float, sigma: float, prior: float | None = None) -> None:
        """One micro-step's loss (a float already synced for the progress bar) and its sigma."""
        if not math.isfinite(loss):
            self.bad_in_a_row += 1
            logger.error("[monitor] the loss is %s (%d in a row)", loss, self.bad_in_a_row)
            if self.bad_in_a_row >= self.stop_after:
                raise NonFiniteLossError(
                    f"the loss was NaN / inf {self.bad_in_a_row} steps in a row; stopping without saving, so the "
                    "last checkpoint and resume state stay as they were. Resume from the state with a lower rate."
                )
            return
        self.bad_in_a_row = 0
        self.last_loss = loss
        self.losses.append(loss)
        for name, lo, hi in BANDS:
            if lo <= sigma < hi:
                self.bands[name].append(loss)
                break
        if prior is not None and math.isfinite(prior):
            self.priors.append(prior)

    def exclude(self, seconds: float) -> None:
        """Time spent outside training (preview rounds), kept out of s/step."""
        self.excluded += float(seconds)

    def step_done(self, step: int, total: int, optimizer: Any = None) -> None:
        """Call once per optimizer step; logs the summary every ``every`` steps."""
        self.steps += 1
        self.steps_total += 1
        if step % self.every and step != total:
            return
        logger.info("[monitor] %s", self.summary(step, total, optimizer))
        self._reset()

    def snapshot(self) -> dict[str, Any]:
        """The window since the last summary as numbers (the control API's stats): mean loss, the loss per noise
        band, the prior's mean, seconds per optimizer step. A key is None until the window has data for it."""
        elapsed = time.time() - self.started - self.excluded
        return {
            "window_steps": self.steps,
            "last_loss": self.last_loss,
            "window_loss": _mean(self.losses),
            "window_loss_bands": {name: sum(values) / len(values) for name, values in self.bands.items() if values},
            "window_prior": _mean(self.priors),
            "seconds_per_step": elapsed / self.steps if self.steps else None,
        }

    def summary(self, step: int, total: int, optimizer: Any = None) -> str:
        parts = [f"step {step}/{total}"]
        if self.losses:
            bands = ", ".join(
                f"{name} {sum(v) / len(v):.3f} n={len(v)}" for name, values in self.bands.items() if (v := values)
            )
            parts.append(f"loss {sum(self.losses) / len(self.losses):.4f} ({bands})")
        if self.priors:
            parts.append(f"prior {sum(self.priors) / len(self.priors):.5f}")
        inner = getattr(optimizer, "optimizer", optimizer)
        if hasattr(inner, "pop_stats"):
            stats = inner.pop_stats()
            if stats["signal"]:
                parts.append("signal " + " ".join(f"{k} {v:+.2f}" for k, v in stats["signal"].items()))
            parts.append(f"nonfinite {stats['nonfinite']}")
            if stats["nonfinite"]:
                logger.warning(
                    "[monitor] %d NaN / inf gradient values since the last summary (zeroed)", stats["nonfinite"]
                )
        if self.drift is not None:
            parts.append("drift " + self.drift.summary(self.steps_total))
        elapsed = time.time() - self.started - self.excluded
        if self.steps:
            parts.append(f"{elapsed / self.steps:.2f} s/step")
        if torch.cuda.is_available():
            parts.append(f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
        return " | ".join(parts)
