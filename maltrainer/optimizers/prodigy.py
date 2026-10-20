"""Prodigy from a trainer's ``train`` config section."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import torch


def build_prodigy(
    params: Iterable[torch.nn.Parameter], train_config: dict[str, Any], lr: float, weight_decay: float
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Prodigy over ``params`` with the ``prodigy_*`` settings of ``train_config``, and the settings used.

    Only what the installed prodigyopt accepts is passed (its signature has changed between releases),
    and settings that are None are left to its defaults. ``train.learning_rate`` is taken as Prodigy's
    scale only when it is at least 0.01; below that, the trainer's usual small rates would starve it, so 1.0
    is used unless ``train.prodigy_lr`` says otherwise.
    """
    from prodigyopt import Prodigy

    requested = {
        "lr": float(train_config.get("prodigy_lr", lr if lr >= 0.01 else 1.0)),
        "betas": (float(train_config.get("adam_beta1", 0.9)), float(train_config.get("prodigy_beta2", 0.999))),
        "beta3": train_config.get("prodigy_beta3"),
        "weight_decay": weight_decay,
        "eps": float(train_config.get("prodigy_eps", 1e-8)),
        "d0": float(train_config.get("prodigy_d0", 1e-6)),
        "d_coef": float(train_config.get("prodigy_d_coef", 1.0)),
        "growth_rate": float(train_config.get("prodigy_growth_rate", float("inf"))),
        "decouple": bool(train_config.get("prodigy_decouple", True)),
        "use_bias_correction": bool(train_config.get("prodigy_use_bias_correction", False)),
        "safeguard_warmup": bool(train_config.get("prodigy_safeguard_warmup", False)),
        "slice_p": int(train_config.get("prodigy_slice_p", 1)),
    }
    supported = set(inspect.signature(Prodigy.__init__).parameters)
    kwargs = {k: v for k, v in requested.items() if k in supported and v is not None}
    return Prodigy(params, **kwargs), kwargs
