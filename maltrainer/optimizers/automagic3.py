"""
Automagic v3 — Ostris's self-adjusting optimizer, vendored from AI-Toolkit / Fizgig (MIT License).

Split by responsibility: ``automagic3_update`` (one parameter's update), ``automagic3_ops`` (sign bits,
stochastic rounding), ``automagic3_stats`` (the group vote signal), ``automagic3_resume`` (load_state_dict)
and ``automagic3_base`` (the shared attributes); ``Automagic3`` below keeps construction, hooks and ``step``.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch

from maltrainer.optimizers.automagic3_resume import StateRestore
from maltrainer.optimizers.automagic3_stats import GroupStats
from maltrainer.optimizers.automagic3_update import ParamUpdate

__all__ = ["Automagic3"]


class Automagic3(ParamUpdate, GroupStats, StateRestore):
    """
    Automagic v3.

    A single learning rate is kept per param group (typically: one lr for
    the whole run). The control principle: the lr RISES while elements hold
    a decisively consistent update direction at the current step size, FALLS
    while their signs decisively alternate (the overshoot signature: weights
    hopping across a minimum flip sign step to step -- shrinking the step is
    what makes a trajectory reappear at a finer scale), and HOLDS on
    everything in between, which is treated as noise.

    Each element keeps a window of its last H (= ``polarity_history``,
    default 4 or 8) update sign bits ("is the update positive", 1-bit packed) --
    H/8 bytes per element, the only per-element optimizer state.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-6,
        min_lr: float = 1e-8,
        max_lr: float = 1e3,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history: int = 8,
        fused: bool = False,
    ) -> None:
        if min_lr > max_lr:
            raise ValueError(f"min_lr ({min_lr}) must be <= max_lr ({max_lr})")
        if lr > 1e-3:
            logging.getLogger(__name__).warning("start lr %s is high; the controller will correct it", lr)
        defaults = {
            "lr": lr,
            "min_lr": min_lr,
            "max_lr": max_lr,
            "beta2": beta2,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "weight_decay": weight_decay,
            "polarity_history": max(2, min(64, int(polarity_history))),
        }
        self.fused = fused  # set before super().__init__, which calls add_param_group per group
        self._built = []
        self._hook_handles = []
        super().__init__(params, defaults)

        # Multiplies every update's step size (not the learned rates): regularisation images train
        # at a fraction of the step (maltrainer/finetune/regularisation.py). 1 = off.
        self.lr_scale = 1.0
        # Training-health statistics, summed on the device (no sync per step) and read with
        # pop_stats(): each group's vote signal, and how many gradient values were NaN / inf
        # (they are zeroed before the update, so without this count they would pass unseen).
        self._signal_sum = {}
        self._signal_steps = {}
        self._nonfinite = None

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        logging.getLogger(__name__).info("Total training parameters: %s", f"{total:,}")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        # As built, overrides included: a reload restores these, not the previous run's saved ones.
        self._built.append({k: group[k] for k in self.defaults})
        self._rebuild_group_index()
        for p in group["params"]:
            if not p.requires_grad:
                continue
            if self.fused:
                # Fused: update each param the moment its grad is ready.
                self._hook_handles.append(p.register_post_accumulate_grad_hook(self._make_backward_hook(group)))
            elif p.dtype != torch.float32:
                # Non-fused: .step() updates; here low-precision grads accumulate stochastically across
                # micro-batches, so repeated round-to-nearest doesn't drop small ones (fp32 is lossless).
                self._hook_handles.append(p.register_post_accumulate_grad_hook(self._make_accum_hook()))

    def _make_accum_hook(self) -> Callable[[Any], None]:
        def _hook(p: Any) -> None:  # Any: _accum_grad is an attribute set on the parameter
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _make_backward_hook(self, group: dict[str, Any]) -> Callable[[torch.Tensor], None]:
        def _hook(p: torch.Tensor) -> None:
            self._update_param(p, group)

        return _hook

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    self._update_param(p, group)
        self._apply_group_votes()
        return loss
