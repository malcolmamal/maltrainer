"""A two-block model in H3 Image's layout whose block Linears are INT8 ConvRot, for the INT8 window rotator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from maltrainer.finetune.rotation_int8 import Int8WindowRotator, Window
from maltrainer.models.minimax_h3_image.finetune_int8 import encode, make_swappable

GROUP = 4
LEAVES = ("attn.qkv_proj", "mlp.fc1")
SHAPES = {"attn.qkv_proj": (6, 8), "mlp.fc1": (4, 8)}  # (out, in); in divisible by the group size


def int8_linear(weight: torch.Tensor) -> nn.Linear:
    """A Linear holding ``weight`` as INT8 ConvRot codes, set up as the loader leaves it."""
    codes, scale = encode(weight, GROUP)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = nn.Parameter(codes, requires_grad=False)
    linear.register_buffer("scale_weight", scale)
    linear.register_buffer("int8_convrot_groupsize", torch.tensor(GROUP, dtype=torch.int32))
    return linear


class Attn(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.qkv_proj = int8_linear(weight)


class Mlp(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.fc1 = int8_linear(weight)


class Block(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.attn = Attn(torch.randn(SHAPES["attn.qkv_proj"], generator=generator))
        self.mlp = Mlp(torch.randn(SHAPES["mlp.fc1"], generator=generator))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(1), Block(2)])
        self.head = nn.Linear(4, 2)  # dense: never rotates
        self.keys = make_swappable(self)


def windows(model: Model) -> list[Window]:
    """One window per block."""
    found = model.keys
    return [(f"block {i}", [k for k in found if k.startswith(f"blocks.{i}.")]) for i in range(2)]


class Optimizers:
    """make_optimizer: SGD with momentum, remembering what it was asked for."""

    def __init__(self) -> None:
        self.made: list[tuple[str, list[nn.Parameter], torch.optim.Optimizer]] = []

    def __call__(self, params: list[nn.Parameter], name: str) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(params, lr=0.5, momentum=0.9)
        self.made.append((name, params, optimizer))
        return optimizer


def rotator(
    model: Model | None = None,
    plan: Sequence[Window] | None = None,
    make_optimizer: Optimizers | None = None,
    dtype: torch.dtype = torch.float32,
) -> Int8WindowRotator:
    model = model or Model()
    return Int8WindowRotator(model, plan or windows(model), torch.device("cpu"), dtype, make_optimizer or Optimizers())
