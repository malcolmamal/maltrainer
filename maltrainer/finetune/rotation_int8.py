"""Full finetuning by rotating windows over an INT8 ConvRot model (the H3 Image finetune).

The frozen model stays on the GPU as INT8 codes (21 GB for H3 Image). One window of Linears at a time is turned into
a trainable bf16 weight, trained for a number of steps, then quantized back to INT8 codes. After Fizgig's H3 finetune
(an INT8 ConvRot source, the master dequantised from the codes, stochastic rounding into bf16), and as the Krea 2
rotation (``rotation.py``), with these differences:

- a Linear is changed in place (``models.minimax_h3_image.finetune_int8``), so a training adapter's wrapper survives
- the master copy is lazy: an untouched weight is still its file's INT8 codes, and is decoded when its window first
  comes up. ``master`` holds the bf16 value of every weight a window has trained
- each window keeps its optimizer state in CPU RAM while frozen, so a rotation does not reset it and a resume
  continues it exactly
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn

from maltrainer.finetune.rotation_util import _remove_hooks, _to_cpu
from maltrainer.models.minimax_h3_image.finetune_int8 import decode, encode, freeze, group_size, unfreeze

Window = tuple[str, list[str]]  # (name, the weight keys it trains, in the model's module order)


def _free_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Int8WindowRotator:
    def __init__(
        self,
        model: nn.Module,
        windows: Sequence[Window],
        device: torch.device,
        dtype: torch.dtype,
        make_optimizer: Callable[[list[nn.Parameter], str], torch.optim.Optimizer],
    ):
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.make_optimizer = make_optimizer
        self.windows = [(name, list(keys)) for name, keys in windows]
        owner: dict[str, str] = {}
        for name, keys in self.windows:
            for key in keys:
                if key in owner:
                    raise ValueError(f"{key} is in two finetune windows, {owner[key]!r} and {name!r}")
                owner[key] = name
        self.master: dict[
            str, torch.Tensor
        ] = {}  # key -> CPU, in the compute dtype (bf16), every weight a window has trained so far
        self.current: int | None = None
        self.optimizer: Any = None  # the active window's optimizer
        self.optimizer_states: dict[int, dict[str, Any]] = {}  # frozen windows', on the CPU

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.windows]

    def keys(self) -> list[str]:
        return [key for _, keys in self.windows for key in keys]

    def _linear(self, key: str) -> nn.Linear:
        return cast(nn.Linear, self.model.get_submodule(key[: -len(".weight")]))

    def window_parameters(self, index: int) -> int:
        return sum(self._linear(key).weight.numel() for key in self.windows[index][1])

    # ------------------------------------------------------------------ rotation

    def activate(self, index: int) -> None:
        """Make window ``index`` trainable (freezing the active one first)."""
        if index == self.current:
            return
        self.deactivate()
        name, keys = self.windows[index]
        params = []
        for key in keys:
            linear, trained = self._linear(key), self.master.get(key)
            weight = decode(linear, self.dtype) if trained is None else trained.to(self.device, self.dtype)
            params.append(unfreeze(linear, weight))
        _free_cache(self.device)
        self.optimizer = self.make_optimizer(params, name)
        saved = self.optimizer_states.pop(index, None)
        if saved is not None:
            self.optimizer.load_state_dict(saved)
        self.current = index

    def deactivate(self) -> None:
        """Write the active window back to the master and quantize it to INT8 again."""
        if self.current is None:
            return
        self.optimizer_states[self.current] = _to_cpu(self.optimizer.state_dict())
        _remove_hooks(self.optimizer)
        self.optimizer = None
        for key in self.windows[self.current][1]:
            linear = self._linear(key)
            weight = linear.weight.detach()
            linear.weight.grad = None
            self.master[key] = weight.to("cpu", self.dtype, copy=True)
            freeze(linear, weight)
        self.current = None
        _free_cache(self.device)

    # ------------------------------------------------------------------ weights

    def active(self, key: str) -> bool:
        return self.current is not None and key in self.windows[self.current][1]

    def weight(self, key: str) -> torch.Tensor:
        """A rotating weight as trained so far, on the CPU in the compute dtype (the active window read live)."""
        if self.active(key):
            return self._linear(key).weight.detach().to("cpu", self.dtype)
        trained = self.master.get(key)
        return decode(self._linear(key), self.dtype).to("cpu") if trained is None else trained

    def int8(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A rotating weight's INT8 codes and row scale on the CPU, as the next forward pass has them."""
        linear = self._linear(key)
        if self.active(key):
            codes, scale = encode(linear.weight.detach(), group_size(linear))
        else:
            codes, scale = linear.weight.detach(), cast(torch.Tensor, linear.scale_weight)
        return codes.to("cpu"), scale.to("cpu")

    def touched(self) -> dict[str, torch.Tensor]:
        """Every weight a window has trained, as it is now (the active window read live): the values a resume needs."""
        trained = dict(self.master)
        if self.current is not None:
            trained.update({key: self.weight(key) for key in self.windows[self.current][1]})
        return trained

    def restore(self, trained: dict[str, torch.Tensor]) -> None:
        """Before the first activate(): put weights a resumed run had trained back, quantized as they were."""
        if self.current is not None:
            raise RuntimeError("restore the trained weights before activating a window")
        for key, value in trained.items():
            linear = self._linear(key)
            freeze(linear, value.to(linear.weight.device, self.dtype))
            self.master[key] = value.to("cpu", self.dtype, copy=True)

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict[str, Any]:
        """Every window's optimizer state (the active one copied), for a resume."""
        states = dict(self.optimizer_states)
        if self.current is not None:
            states[self.current] = _to_cpu(self.optimizer.state_dict())
        return {"windows": self.names, "optimizers": states}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Before the first activate(): the saved per-window optimizer states."""
        if state.get("windows") != self.names:
            raise ValueError(
                f"The resumed run rotated windows {state.get('windows')}, this config has {self.names}; "
                "keep finetune.blocks_per_window, finetune.block_range and finetune.targets as they were"
            )
        if self.current is not None:
            raise RuntimeError("load the optimizer states before activating a window")
        self.optimizer_states = {int(k): v for k, v in state["optimizers"].items()}
