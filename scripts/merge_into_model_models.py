"""The merge loop of scripts/merge_lora_into_model.py: other checkpoints blended into the base, then the LoRAs.

Per tensor of the base A, in float32 on the chosen device:

    W = (1 - sum t_i) * A + sum t_i * B_i          weighted sum, ``models:``
        + sum s_j * (B_j - C_j)                     add difference, ``add_difference:`` (C_j = the model B_j came from)
        + the LoRAs                                 merge_into_model_core.py

A tensor a model lacks, or holds in another shape, keeps the base's value for that term; models that look unrelated
(far from the base, or B far from its C), fp8 inputs to a difference and weights outside [0, 1] are reported as
warnings, not refused: the output is still written, and the warnings say what to check.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from tqdm import tqdm

from merge_into_model_core import (
    Deltas,
    add_lora,
    bare,
    check_lora_modules,
    check_model,
    is_plain_float,
    lora_parts,
    read_float,
)

FAR = 0.25  # relative distance above which two models look like different bases (a finetune moves ~1-5%)


@dataclass
class Model:
    """One opened checkpoint: its handle, keys by bare name, and the distance it accumulates."""

    path: Path
    handle: Any
    all_keys: set[str]
    keys: dict[str, str]  # bare name -> stored name, the floating tensors only
    missing: set[str] = field(default_factory=set)
    reshaped: set[str] = field(default_factory=set)

    @property
    def is_fp8(self) -> bool:
        return any(k.endswith(".weight_scale") for k in self.all_keys)

    def get(self, bare_key: str, like: torch.Tensor) -> torch.Tensor | None:
        """This model's tensor in float32 on ``like``'s device; None (and noted) if it lacks it or has another shape."""
        key = self.keys.get(bare_key)
        if key is None:
            self.missing.add(bare_key)
            return None
        tensor = read_float(self.handle, self.all_keys, key)
        if tensor.shape != like.shape:
            self.reshaped.add(bare_key)
            return None
        return tensor.to(like.device)


@dataclass
class Terms:
    weighted: list[tuple[Path, float]] = field(default_factory=list)
    differences: list[tuple[Path, Path, float]] = field(default_factory=list)  # (B, C, strength)


@contextlib.contextmanager
def opened(paths: list[Path]) -> Iterator[dict[Path, Model]]:
    with contextlib.ExitStack() as stack:
        models = {}
        for path in dict.fromkeys(paths):
            handle = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            keys = set(handle.keys())
            check_model(path, handle, keys)
            models[path] = Model(path, handle, keys, {bare(k): k for k in keys if is_plain_float(handle, k)})
        yield models


class Distance:
    """Accumulated ||x - y|| / ||y|| per (label) over every merged tensor."""

    def __init__(self) -> None:
        self.num: dict[str, float] = {}
        self.den: dict[str, float] = {}

    def add(self, label: str, x: torch.Tensor, y: torch.Tensor) -> None:
        self.num[label] = self.num.get(label, 0.0) + float((x - y).pow(2).sum())
        self.den[label] = self.den.get(label, 0.0) + float(y.pow(2).sum())

    def value(self, label: str) -> float:
        return float((self.num.get(label, 0.0) / max(self.den.get(label, 0.0), 1e-30)) ** 0.5)


def _blend(base: torch.Tensor, name: str, terms: Terms, models: dict[Path, Model], dist: Distance) -> torch.Tensor:
    out = base * (1.0 - sum(t for _, t in terms.weighted))
    for path, t in terms.weighted:
        other = models[path].get(name, base)
        out = out + t * (base if other is None else other)
        if other is not None:
            dist.add(path.name, other, base)
    for b, c, s in terms.differences:
        tb, tc = models[b].get(name, base), models[c].get(name, base)
        if tb is not None and tc is not None:
            out = out + s * (tb - tc)
            dist.add(f"{b.name} - {c.name}", tb, tc)
    return out


def _warnings(terms: Terms, models: dict[Path, Model], base: Path, dist: Distance) -> list[str]:
    base_keys = set(models[base].keys)
    out = []
    total = sum(t for _, t in terms.weighted)
    if terms.weighted and not 0.0 <= total <= 1.0:
        out.append(
            f"the model weights sum to {total:g}: outside [0, 1] the base's own share is negative (extrapolation)"
        )
    for path, model in ((p, m) for p, m in models.items() if p != base):
        extra = set(model.keys) - base_keys
        for what, names in (("lacks", model.missing), ("holds in another shape", model.reshaped), ("adds", extra)):
            if names:
                kept = "ignored" if what == "adds" else "the base's value is kept there"
                out.append(f"{path.name} {what} {len(names)} tensors ({kept}), e.g. {sorted(names)[:3]}")
    for path, _ in terms.weighted:
        if dist.value(path.name) > FAR:
            out.append(f"{path.name} is {dist.value(path.name):.0%} away from the base: a different base model?")
    for b, c, _ in terms.differences:
        label = f"{b.name} - {c.name}"
        if dist.value(label) > FAR:
            out.append(f"{b.name} is {dist.value(label):.0%} away from {c.name}: is it really a finetune of it?")
        fp8 = [p.name for p in (b, c) if models[p].is_fp8]
        if fp8:
            out.append(f"{', '.join(fp8)} is fp8: its rounding noise goes into {label} (bf16 / fp16 is cleaner)")
    return out


def merge_into(
    base: Path, terms: Terms, deltas: Deltas, dtype: torch.dtype, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    """(the merged tensors, the base's metadata without its fp8 quantization record, the report with warnings)."""
    others = [p for p, _ in terms.weighted] + [p for b, c, _ in terms.differences for p in (b, c)]
    out: dict[str, torch.Tensor] = {}
    changes, dist = [], Distance()
    with opened([base, *others]) as models:
        handle = models[base].handle
        keys = set(handle.keys())
        check_lora_modules(base, keys, deltas)
        metadata = {k: v for k, v in (handle.metadata() or {}).items() if k != "_quantization_metadata"}
        for key in tqdm(sorted(keys), desc=f"Merging into {base.name}", unit="tensor"):
            if key.endswith(".weight_scale"):
                continue  # folded into its dequantized weight
            if not is_plain_float(handle, key):
                out[key] = handle.get_tensor(key)
                continue
            original = read_float(handle, keys, key)
            tensor = _blend(original.to(device), bare(key), terms, models, dist).cpu() if others else original
            parts = lora_parts(deltas, key)
            if parts:
                tensor = add_lora(tensor, key, parts, device)
            if others or parts:
                changes.append(float((tensor - original).norm() / original.norm().clamp_min(1e-12)))
            out[key] = tensor.to(dtype).contiguous()
        warnings = _warnings(terms, models, base, dist)
    report = {
        "merged_tensors": len(changes),
        "mean_change": sum(changes) / max(len(changes), 1),
        "max_change": max(changes, default=0.0),
        "distances": {label: round(dist.value(label), 4) for label in dist.num},
        "warnings": warnings,
    }
    return out, metadata, report
