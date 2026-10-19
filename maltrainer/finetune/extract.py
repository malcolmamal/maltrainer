"""
A LoRA from a finetune: the per-layer difference between the finetuned and the base checkpoint,
cut down to low rank with an SVD (kohya's extract_lora_from_models, LyCORIS's extract_locon).

For each 2-D weight W that changed, D = W_tuned - W_base (in fp32) is factored D = U S V^T, and the
top k singular directions become lora_B = U_k sqrt(S_k), lora_A = sqrt(S_k) V_k^T, alpha = k
(scale 1). One SVD per layer serves every requested rank cap, so several files cost one pass.

How many singular values each layer keeps (``mode``, LyCORIS's names where they exist):

- ``fixed``: exactly the rank cap
- ``fro``: the fewest that hold ``value`` of the layer's Frobenius norm (kohya's sv_fro; LyCORIS's
  ``quantile``), at most the cap
- ``ratio``: those with s_i >= ``value`` * s_max, at most the cap
- ``threshold``: those with s_i >= ``value``, at most the cap

The dynamic modes give layers the finetune barely touched a small rank and heavily changed ones
more. Each layer's dropped share of ||D|| is reported, so the cost of a cap is visible.

Model-specific: the ExtractProfile says which tensors are LoRA targets and how a stored weight
splits into the modules a LoRA names (e.g. a fused qkv into q / k / v). The default takes every
2-D ``.weight`` as one module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

CONTAINER_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
MODES = ("fixed", "fro", "ratio", "threshold")

# module name (no ".weight"), diff [out, in] -> [(LoRA module name, diff part)]
SplitFn = Callable[[str, torch.Tensor], list[tuple[str, torch.Tensor]]]


def _one_module(name: str, diff: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    return [(name, diff)]


def _is_linear_weight(name: str, shape: Sequence[int]) -> bool:  # noqa: ARG001 - a profile's predicate signature
    return len(shape) == 2


@dataclass(frozen=True)
class ExtractProfile:
    name: str = "generic"
    # which stored tensors (module name, shape) become LoRA modules
    include: Callable[[str, Sequence[int]], bool] = _is_linear_weight
    split: SplitFn = _one_module
    # whether a checkpoint (its module names, container prefix stripped) is this model's
    detect: Callable[[Iterable[str]], bool] = field(default=lambda _keys: False)


GENERIC = ExtractProfile()


def strip_prefix(keys: list[str]) -> str:
    for prefix in CONTAINER_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return prefix
    return ""


def keep_count(s: torch.Tensor, mode: str, value: float, cap: int) -> int:
    """How many of the (descending) singular values ``s`` to keep."""
    n = s.numel()
    cap = max(1, min(int(cap), n))
    if mode == "fixed":
        return cap
    if float(s[0]) <= 0.0:
        return 1
    if mode == "fro":
        energy = s.double() ** 2
        share = energy.cumsum(0) / energy.sum()
        k = int(torch.searchsorted(share, torch.tensor(float(value), dtype=torch.float64)).item()) + 1
    elif mode == "ratio":
        k = int((s >= float(value) * s[0]).sum().item())
    elif mode == "threshold":
        k = int((s >= float(value)).sum().item())
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return max(1, min(k, cap))


def factor(diff: torch.Tensor, caps: Sequence[int], mode: str, value: float, device: str):
    """{cap: (down [k, in], up [out, k], k, dropped share of ||diff||)} from one SVD, in fp32 on the CPU."""
    u, s, vh = torch.linalg.svd(diff.to(device, torch.float32), full_matrices=False)
    energy = s.double() ** 2
    total = float(energy.sum())
    out = {}
    for cap in caps:
        k = keep_count(s, mode, value, cap)
        root = s[:k].sqrt()
        up = (u[:, :k] * root[None, :]).cpu()
        down = (root[:, None] * vh[:k]).cpu()
        error = (float(energy[k:].sum()) / total) ** 0.5 if total > 0 else 0.0
        out[cap] = (down, up, k, error)
    return out


def _check_same_tensors(base, tuned, base_names: dict[str, str], tuned_names: dict[str, str]) -> None:
    """Both checkpoints hold the same tensors (names without their container prefix) in the same shapes."""
    if set(base_names) != set(tuned_names):
        only = sorted(set(base_names) ^ set(tuned_names))
        raise ValueError(f"the checkpoints have different tensors, e.g. {only[:3]} ({len(only)} differ)")
    reshaped = sorted(
        name
        for name in base_names
        if base.get_slice(base_names[name]).get_shape() != tuned.get_slice(tuned_names[name]).get_shape()
    )
    if reshaped:
        raise ValueError(f"the checkpoints' tensors differ in shape, e.g. {reshaped[:3]} ({len(reshaped)} differ)")


def extract(
    base_path: str | Path,
    tuned_path: str | Path,
    caps: Sequence[int],
    mode: str = "fixed",
    value: float = 0.0,
    profile: ExtractProfile | None = None,
    device: str = "cpu",
    out_prefix: str = "diffusion_model.",
    dtype: torch.dtype = torch.bfloat16,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    """({cap: LoRA state dict}, report). Both files must have the same keys and shapes."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    caps = sorted({int(c) for c in caps})
    states: dict[int, dict[str, torch.Tensor]] = {cap: {} for cap in caps}
    ranks: dict[int, list[int]] = {cap: [] for cap in caps}
    errors: dict[int, list[float]] = {cap: [] for cap in caps}
    unchanged, skipped_changed = 0, []
    with (
        safe_open(str(base_path), framework="pt", device="cpu") as base,
        safe_open(str(tuned_path), framework="pt", device="cpu") as tuned,
    ):
        base_keys, tuned_keys = list(base.keys()), list(tuned.keys())
        base_prefix, tuned_prefix = strip_prefix(base_keys), strip_prefix(tuned_keys)
        base_names = {k[len(base_prefix) :]: k for k in base_keys}
        tuned_names = {k[len(tuned_prefix) :]: k for k in tuned_keys}
        _check_same_tensors(base, tuned, base_names, tuned_names)
        profile = profile or GENERIC
        names = [n for n in base_names if n.endswith(".weight")]
        iterator = progress(names) if progress else names
        for name in iterator:
            b = base.get_slice(base_names[name])
            shape, stored = b.get_shape(), b.get_dtype()
            if stored in ("I8", "U8", "F8_E4M3", "F8_E5M2"):
                raise ValueError(f"{name} is stored as {stored}; extraction needs plain (bf16/fp16/fp32) checkpoints")
            w_base = base.get_tensor(base_names[name])
            w_tuned = tuned.get_tensor(tuned_names[name])
            if torch.equal(w_base, w_tuned):
                unchanged += 1
                continue
            module = name[: -len(".weight")]
            if not profile.include(module, shape):
                skipped_changed.append(module)
                continue
            diff = w_tuned.float() - w_base.float()
            for part, part_diff in profile.split(module, diff):
                if not torch.any(part_diff):
                    continue
                for cap, (down, up, k, error) in factor(part_diff, caps, mode, value, device).items():
                    key = f"{out_prefix}{part}"
                    states[cap][f"{key}.lora_A.weight"] = down.to(dtype).contiguous()
                    states[cap][f"{key}.lora_B.weight"] = up.to(dtype).contiguous()
                    states[cap][f"{key}.alpha"] = torch.tensor(float(k), dtype=torch.float32)
                    ranks[cap].append(k)
                    errors[cap].append(error)
    if not any(ranks.values()):
        raise ValueError("no LoRA target changed between the two checkpoints")
    report = {
        "profile": profile.name,
        "mode": mode,
        "value": value,
        "unchanged_tensors": unchanged,
        "changed_but_not_extracted": skipped_changed,
        "per_cap": {
            cap: {
                "modules": len(ranks[cap]),
                "rank_min": min(ranks[cap]),
                "rank_max": max(ranks[cap]),
                "rank_mean": sum(ranks[cap]) / len(ranks[cap]),
                "dropped_mean": sum(errors[cap]) / len(errors[cap]),
                "dropped_worst": max(errors[cap]),
            }
            for cap in caps
        },
    }
    return states, report
