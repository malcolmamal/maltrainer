"""A two-block model with the ``blocks.N.<leaf>`` layout WindowRotator rotates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from maltrainer.finetune.rotation import Window, WindowRotator

# every row is (m, m / 2): fp8 holds it exactly once the row maximum is scaled to 448
ATTN = torch.tensor([[1.0, 0.5], [-2.0, -1.0]])
MLP = torch.tensor([[1.0, 0.5], [4.0, 2.0], [-1.0, -0.5]])
WINDOWS: Sequence[Window] = (("attn", ("attn.",)), ("mlp", ("mlp",)))


class Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(2, 2, bias=False)
        self.wk = nn.Linear(2, 2, bias=False)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attn()
        self.mlp = nn.Linear(2, 3)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(2, 2)


def master() -> dict[str, torch.Tensor]:
    out = {}
    for i in range(2):
        out[f"blocks.{i}.attn.wq.weight"] = ATTN.to(torch.bfloat16)
        out[f"blocks.{i}.attn.wk.weight"] = (ATTN * 2).to(torch.bfloat16)
        out[f"blocks.{i}.mlp.weight"] = MLP.to(torch.bfloat16)
    return out


class Optimizers:
    """make_optimizer: SGD with momentum, remembering what it was asked for."""

    def __init__(self) -> None:
        self.made: list[tuple] = []

    def __call__(self, params: list[nn.Parameter], name: str) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(params, lr=0.5, momentum=0.9)
        self.made.append((name, params, optimizer))
        return optimizer


def rotator(windows: Sequence[Window] = WINDOWS, make_optimizer=None) -> WindowRotator:
    return WindowRotator(Model(), master(), windows, "cpu", torch.float32, make_optimizer or Optimizers())


def step(windows: WindowRotator) -> None:
    """One SGD step on the active window, so its optimizer has momentum state."""
    loss = sum((param**2).sum() for group in windows.optimizer.param_groups for param in group["params"])
    loss.backward()
    windows.optimizer.step()
