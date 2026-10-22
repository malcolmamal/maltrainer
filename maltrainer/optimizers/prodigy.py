"""Prodigy from a trainer's ``train`` config section."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)


def build_prodigy(
    params: Iterable[torch.nn.Parameter],
    train_config: dict[str, Any],
    lr: float,
    weight_decay: float,
    defaults: Mapping[str, Any] | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Prodigy over ``params`` with the ``prodigy_*`` settings of ``train_config``, and the settings used.

    Only what the installed prodigyopt accepts is passed (its signature has changed between releases; a
    setting it lacks is logged as ignored), and settings that are None are left to its defaults.
    ``train.learning_rate`` is taken as Prodigy's scale only when it is at least 0.01; below that, the
    trainer's usual small rates would starve it, so 1.0 is used unless ``train.prodigy_lr`` says otherwise.

    ``defaults`` replaces the value used when a ``prodigy_*`` key is absent (``prodigy_beta2``, ``prodigy_d_coef``,
    ``prodigy_use_bias_correction``, ``prodigy_safeguard_warmup``, ``prodigy_slice_p``; None drops a setting).
    """
    from prodigyopt import Prodigy

    given = {
        "prodigy_beta2": 0.999,
        "prodigy_d_coef": 1.0,
        "prodigy_use_bias_correction": False,
        "prodigy_safeguard_warmup": False,
        "prodigy_slice_p": 1,
        **(defaults or {}),
    }
    slice_p = train_config.get("prodigy_slice_p", given["prodigy_slice_p"])
    requested = {
        "lr": float(train_config.get("prodigy_lr", lr if lr >= 0.01 else 1.0)),
        "betas": (
            float(train_config.get("adam_beta1", 0.9)),
            float(train_config.get("prodigy_beta2", given["prodigy_beta2"])),
        ),
        "beta3": train_config.get("prodigy_beta3"),
        "weight_decay": weight_decay,
        "eps": float(train_config.get("prodigy_eps", 1e-8)),
        "d0": float(train_config.get("prodigy_d0", 1e-6)),
        "d_coef": float(train_config.get("prodigy_d_coef", given["prodigy_d_coef"])),
        "growth_rate": float(train_config.get("prodigy_growth_rate", float("inf"))),
        "decouple": bool(train_config.get("prodigy_decouple", True)),
        "use_bias_correction": bool(
            train_config.get("prodigy_use_bias_correction", given["prodigy_use_bias_correction"])
        ),
        "safeguard_warmup": bool(train_config.get("prodigy_safeguard_warmup", given["prodigy_safeguard_warmup"])),
        "slice_p": None if slice_p is None else int(slice_p),
    }
    supported = set(inspect.signature(Prodigy.__init__).parameters)
    kwargs = {k: v for k, v in requested.items() if k in supported and v is not None}
    dropped = sorted(set(requested) - set(kwargs) - {"beta3"})
    if dropped:
        logger.warning("Installed prodigyopt does not accept %s; ignored.", dropped)
    return Prodigy(params, **kwargs), kwargs
