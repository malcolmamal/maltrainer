"""
Full finetuning by rotating windows, for models too big to train every weight at once.

The frozen model stays on the GPU in fp8 (``Fp8ScaledLinear``). One window of Linears at a time
(e.g. every block's attention, then every block's MLP gate) is swapped up to a trainable bf16
``nn.Linear`` from a CPU master copy, trained for a number of steps, then written back to the
master and re-quantized to fp8 for the forward passes of the other windows. A cycle through all
windows trains every rotating weight once. After Fizgig's full finetune (fizgig/families/ft.py),
with two differences:

- the frozen trunk is fp8 per output row, not NF4: closer to the bf16 weights, and it fits a
  32 GB card with a window of a 12.9B model
- each window keeps its optimizer state (moved to CPU RAM while the window is frozen), so a
  rotation does not reset it, and a resume continues it exactly

The master is the source of truth: a window starts from it and writes back to it, so training
never round-trips through fp8. Saves read the master, with the active window laid over it.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear, quantize_rows

logger = logging.getLogger(__name__)

# A window: (name, Linear-name prefixes inside a block). "attn." matches blocks.N.attn.wq etc.
Window = tuple[str, tuple[str, ...]]


def _block_leaf(key: str, blocks: str) -> str | None:
    """``blocks.12.attn.wq.weight`` -> ``attn.wq`` (None outside the numbered blocks)."""
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != blocks or not parts[1].isdigit() or parts[-1] != "weight":
        return None
    return ".".join(parts[2:-1])


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value


def _remove_hooks(optimizer: Any) -> None:
    """Fused optimizers (Automagic3) step from grad hooks; a frozen window's must not fire."""
    for handle in getattr(optimizer, "_hook_handles", []):
        handle.remove()
    if hasattr(optimizer, "_hook_handles"):
        optimizer._hook_handles = []


class WindowRotator:
    """Swaps windows of block Linears between frozen fp8 and trainable bf16, in place.

    ``master``: weight key (``blocks.N.<leaf>.weight``) -> bf16 CPU tensor, every rotating weight.
    ``make_optimizer(params, name)``: a fresh optimizer for a window's params (in a fixed order).
    """

    def __init__(
        self,
        model: nn.Module,
        master: dict[str, torch.Tensor],
        windows: Sequence[Window],
        device: torch.device,
        compute_dtype: torch.dtype,
        make_optimizer: Callable[[list[nn.Parameter], str], torch.optim.Optimizer],
        blocks: str = "blocks",
    ):
        self.model, self.master = model, master
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.make_optimizer = make_optimizer
        # every window's weight keys, in the model's own module order (stable across loads, so an
        # optimizer state saved by index lines up again)
        ordered = [f"{name}.weight" for name, _ in model.named_modules() if f"{name}.weight" in master]
        self.windows: list[tuple[str, list[str]]] = []
        owner: dict[str, str] = {}
        for name, prefixes in windows:
            keys = [k for k in ordered if (_block_leaf(k, blocks) or "").startswith(tuple(prefixes))]
            if not keys:
                raise ValueError(f"finetune window {name!r} ({list(prefixes)}) matches no Linear")
            for key in keys:
                if key in owner:
                    raise ValueError(f"{key} is in two finetune windows, {owner[key]!r} and {name!r}")
                owner[key] = name
            self.windows.append((name, keys))
        untrained = [k for k in ordered if k not in owner]
        if untrained:
            logger.warning(
                "%d block Linears are in no finetune window and stay frozen, e.g. %s", len(untrained), untrained[:3]
            )
        self.current: int | None = None
        self.optimizer: Any = None  # the active window's torch Optimizer
        self.optimizer_states: dict[int, dict] = {}  # frozen windows' optimizer state, on the CPU

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.windows]

    def window_parameters(self, index: int) -> int:
        return sum(self.master[k].numel() for k in self.windows[index][1])

    def _swap(self, key: str, module: nn.Module) -> nn.Module:
        path = key[: -len(".weight")]
        parent_path, _, attribute = path.rpartition(".")
        parent = self.model.get_submodule(parent_path)
        old = getattr(parent, attribute)
        setattr(parent, attribute, module)
        return old

    # ------------------------------------------------------------------ rotation

    def activate(self, index: int) -> None:
        """Make window ``index`` trainable (freezing the active one first)."""
        if index == self.current:
            return
        if self.current is not None:
            self.deactivate()
        name, keys = self.windows[index]
        params: list[nn.Parameter] = []
        for key in keys:
            frozen: Any = self.model.get_submodule(key[: -len(".weight")])
            linear = nn.Linear(frozen.in_features, frozen.out_features, bias=False, device="meta")
            linear.weight = nn.Parameter(self.master[key].to(self.device, self.compute_dtype), requires_grad=True)
            if getattr(frozen, "bias", None) is not None:
                linear.bias = nn.Parameter(frozen.bias.detach(), requires_grad=False)  # biases stay frozen
            self._swap(key, linear)
            params.append(linear.weight)
        del frozen
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.optimizer = self.make_optimizer(params, name)
        saved = self.optimizer_states.pop(index, None)
        if saved is not None:
            self.optimizer.load_state_dict(saved)
        self.current = index

    def deactivate(self) -> None:
        """Write the active window back to the master and freeze it to fp8 again."""
        if self.current is None:
            return
        _, keys = self.windows[self.current]
        self.optimizer_states[self.current] = _to_cpu(self.optimizer.state_dict())
        _remove_hooks(self.optimizer)
        self.optimizer = None
        for key in keys:
            linear: Any = self.model.get_submodule(key[: -len(".weight")])
            weight = linear.weight.detach()
            linear.weight.grad = None
            self.master[key] = weight.to("cpu", torch.bfloat16, copy=True)  # before the lossy fp8 encode
            fp8, scale = quantize_rows(weight)
            with torch.device("meta"):  # the buffers are replaced right away
                frozen = Fp8ScaledLinear(
                    linear.in_features, linear.out_features, linear.bias is not None, self.compute_dtype
                )
            frozen.weight, frozen.weight_scale = fp8, scale
            if linear.bias is not None:
                frozen.bias = linear.bias.detach()
            self._swap(key, frozen)
            del weight, linear
        self.current = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ state

    def weight(self, key: str) -> torch.Tensor:
        """A rotating weight as trained so far, bf16 on the CPU (the active window read live)."""
        if self.current is not None and key in self.windows[self.current][1]:
            return self.model.get_parameter(key).detach().to("cpu", torch.bfloat16)
        return self.master[key]

    def keys(self) -> list[str]:
        return [key for _, keys in self.windows for key in keys]

    def state_dict(self) -> dict:
        """Every window's optimizer state (the active one copied), for a resume."""
        states = dict(self.optimizer_states)
        if self.current is not None:
            states[self.current] = _to_cpu(self.optimizer.state_dict())
        return {"windows": self.names, "optimizers": states}

    def load_state_dict(self, state: dict) -> None:
        """Before the first activate(): the saved per-window optimizer states."""
        if state.get("windows") != self.names:
            raise ValueError(
                f"The resumed run rotated windows {state.get('windows')}, this config has {self.names}; "
                "keep finetune.windows as it was (or continue from the weights with a fresh optimizer)"
            )
        if self.current is not None:
            raise RuntimeError("load the optimizer states before activating a window")
        self.optimizer_states = {int(k): v for k, v in state["optimizers"].items()}
