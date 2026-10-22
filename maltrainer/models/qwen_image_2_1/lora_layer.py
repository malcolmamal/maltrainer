"""The LoRA layer wrapped around one linear projection of the Qwen-Image-2.1 transformer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class QwenImage21LoRALayer(nn.Module):
    """Single LoRA layer for Linear projection in Qwen-Image-2.1."""

    def __init__(self, original_module: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_features = original_module.in_features
        out_features = original_module.out_features

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.lora_down.weight.dtype
        if x.is_floating_point() and x.dtype != target_dtype and not torch.is_autocast_enabled():
            x = x.to(target_dtype)
        return cast(torch.Tensor, self.lora_up(self.lora_down(x)) * self.scale)
