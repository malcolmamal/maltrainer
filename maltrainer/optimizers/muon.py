"""Muon (momentum orthogonalised by Newton-Schulz) for a LoRA's 2D factors.

The update is PyTorch's ``torch.optim.Muon`` (its momentum, quintic Newton-Schulz iteration and learning-rate
adjustment rules; BSD-3-Clause, Copyright (c) the PyTorch contributors), with the per-matrix work batched. A LoRA is
hundreds of small matrices, and a loop of three matmuls per iteration per matrix costs a third of a step: here the
matrices of one shape are orthogonalised together, and the momentum and weight decay run as foreach ops.

With ``adjust_lr_fn="match_rms_adamw"`` a step is as large as AdamW's, so AdamW's learning rates apply (1e-4).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor

NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
ADJUST_LR_FNS = (None, "original", "match_rms_adamw", "spectral_unclamped")


def adjusted_lr(lr: float, adjust_lr_fn: str | None, shape: tuple[int, ...]) -> float:
    """The learning rate for a matrix of ``shape`` (torch's ``_adjust_lr``)."""
    rows, cols = shape[0], shape[1]
    if adjust_lr_fn in (None, "original"):
        return lr * math.sqrt(max(1.0, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    return lr * math.sqrt(rows / cols)


def orthogonalise(stack: Tensor, coefficients: tuple[float, float, float], steps: int, eps: float) -> Tensor:
    """Newton-Schulz on a (count, rows, cols) stack of same-shaped matrices, in bf16 like torch's."""
    a, b, c = coefficients
    x = stack.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
    return x.transpose(-2, -1) if tall else x


class BatchedMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
        eps: float = 1e-7,
    ) -> None:
        if adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(f"adjust_lr_fn {adjust_lr_fn!r} is not one of {ADJUST_LR_FNS}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "eps": eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(f"BatchedMuon takes 2D parameters only, got {tuple(parameter.shape)}")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                self._update_group(group, params)
        return loss

    def _update_group(self, group: dict[str, Any], params: list[Tensor]) -> None:
        lr, momentum = float(group["lr"]), float(group["momentum"])
        grads: list[Tensor] = [cast(Tensor, p.grad) for p in params]
        buffers = [
            self.state[p].setdefault("momentum_buffer", torch.zeros_like(g)) for p, g in zip(params, grads, strict=True)
        ]
        torch._foreach_lerp_(buffers, grads, 1.0 - momentum)
        updates = [g.lerp(b, momentum) for g, b in zip(grads, buffers, strict=True)] if group["nesterov"] else buffers
        if group["weight_decay"]:
            torch._foreach_mul_(params, 1.0 - lr * float(group["weight_decay"]))
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for index, parameter in enumerate(params):
            by_shape.setdefault(tuple(parameter.shape), []).append(index)
        for shape, indices in by_shape.items():
            ortho = orthogonalise(
                torch.stack([updates[i] for i in indices]),
                NS_COEFFICIENTS,
                int(group["ns_steps"]),
                float(group["eps"]),
            )
            step = adjusted_lr(lr, group["adjust_lr_fn"], shape)
            torch._foreach_add_(
                [params[i] for i in indices],
                [u.to(params[i].dtype) for u, i in zip(ortho, indices, strict=True)],
                alpha=-step,
            )
