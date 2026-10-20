"""The LoRA layers of the MiniMax H3 network: the plain one and the AdaLN grid one."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class MiniMaxH3LoRALayer(nn.Module):
    """Single LoRA layer for Linear projection in MiniMax H3."""

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
        # Adapters are held in higher precision than the frozen base (fp32 on a bf16/INT8
        # base) so that optimizer updates accumulate; the matmuls themselves do not need it.
        # Under autocast the linear would recast to bf16 anyway, so upcasting the activation
        # there would only materialize a large fp32 tensor for nothing.
        target_dtype = self.lora_down.weight.dtype
        if x.is_floating_point() and x.dtype != target_dtype and not torch.is_autocast_enabled():
            x = x.to(target_dtype)
        return cast(torch.Tensor, self.lora_up(self.lora_down(x)) * self.scale)


class GridAdaLNLoRA(nn.Module):
    """A full-model AdaLN LoRA row, run on the pruned base's curve-table AdaLN.

    The full model's adaln_proj reads silu(time_embedder(t)) [2688]; the pruned build replaced
    that with an 8-wide table lerped by t, so the full-width input does not exist at run time.
    Each incoming table row is matched to its nearest grid row and the full model's
    precomputed silu(t_emb) at that point stands in -- half a grid step (1/2048 in t) of error
    at worst, the approach of larryvrh's ComfyUI-MiniMax-H3-Turbo node and of Fizgig.
    """

    def __init__(self, down: torch.Tensor, up: torch.Tensor, scale: float, table: torch.Tensor, grid: torch.Tensor):
        super().__init__()
        self.register_buffer("down", down.detach().clone().float())
        self.register_buffer("up", up.detach().clone().float())
        self.register_buffer("table", table.detach().clone().float())
        self.register_buffer("grid", grid.detach().clone().float())
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.cdist(x.detach().float(), self.get_buffer("table").float()).argmin(dim=1)
        full = self.get_buffer("grid")[index].to(self.get_buffer("down").dtype)
        return (full @ self.get_buffer("down").T @ self.get_buffer("up").T) * self.scale
