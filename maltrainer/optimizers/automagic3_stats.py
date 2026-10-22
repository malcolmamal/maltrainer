"""
Automagic v3: the per-group vote signal that moves the learning rates, and the statistics read off it.

Mixed into ``Automagic3`` (see ``automagic3``).
"""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.optimizers.automagic3_base import AutomagicState


class GroupStats(AutomagicState):
    def _apply_group_votes(self) -> None:
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)
            name = group.get("name", str(gi))
            total = self._signal_sum.get(name)
            self._signal_sum[name] = signal.detach().clone() if total is None else total.add_(signal.to(total.device))
            self._signal_steps[name] = self._signal_steps.get(name, 0) + 1
            factor = torch.exp(signal)
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "lr" not in st:
                    continue
                lr_t = st["lr"]
                f = factor if factor.device == lr_t.device else factor.to(lr_t.device)
                lr_t.mul_(f).clamp_(min=group["min_lr"], max=group["max_lr"])
            self._group_num[gi] = None
            self._group_den[gi] = None

    def pop_stats(self) -> dict[str, Any]:
        """Statistics since the last call, then reset (one GPU sync).

        signal: per group, the mean of the vote signal in [-1, 1], weighted by update size: the
            share of elements whose update kept one sign over the last ``polarity_history`` steps
            minus the share whose sign flipped every step. Positive = updates keep a direction
            (learning); near 0 = sign noise; negative = overshooting. Computed even when the rate
            is pinned (min_lr = max_lr), where it no longer moves the rate.
        nonfinite: gradient values that were NaN or inf (zeroed before their update).
        """
        signal = {name: float(total) / self._signal_steps[name] for name, total in self._signal_sum.items()}
        nonfinite = int(self._nonfinite) if self._nonfinite is not None else 0
        self._signal_sum, self._signal_steps, self._nonfinite = {}, {}, None
        return {"signal": signal, "nonfinite": nonfinite}

    def get_learning_rates(self) -> list[float]:
        out = []
        for group in self.param_groups:
            lrs = [self.state[p]["lr"] for p in group["params"] if p in self.state and "lr" in self.state[p]]
            out.append(float(torch.stack(lrs).mean()) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])
