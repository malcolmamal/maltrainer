"""
Automagic v3: the attributes the mixins share and the group index (see ``automagic3``).
"""

from __future__ import annotations

from typing import Any

import torch


class AutomagicState(torch.optim.Optimizer):
    """What ``Automagic3.__init__`` sets up, declared once so the mixins type-check against it."""

    fused: bool
    lr_scale: float
    _built: list[dict[str, Any]]
    _hook_handles: list[Any]
    _signal_sum: dict[str, Any]
    _signal_steps: dict[str, Any]
    _nonfinite: torch.Tensor | None
    _param_group_index: dict[Any, int]
    _group_num: list[Any]
    _group_den: list[Any]

    def _rebuild_group_index(self) -> None:
        self._param_group_index = {p: gi for gi, group in enumerate(self.param_groups) for p in group["params"]}
        self._group_num = [None] * len(self.param_groups)
        self._group_den = [None] * len(self.param_groups)
