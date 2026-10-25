"""The LoRA side of scripts/merge_lora_into_model.py, and the model-file helpers its merge loop
(merge_into_model_models.py) shares: read each LoRA's deltas, add them into a weight, read a model's tensors.

A LoRA adds, per weight, in float32 on the chosen device:

    W' = W + sum_i strength_i * (alpha_i / rank_i) * B_i @ A_i      (a LoRA pair)
           + sum_j strength_j * D_j                                   (a ComfyUI full delta, ``<module>.diff``)
           + sum_k strength_k * scale_k * kron(w1_k, w2_k)            (a LoKr module, merge_into_model_lokr.py)

and ``<module>.diff_b`` adds to the bias the same way. LoRA module names are matched to the base's by dropping the
container prefixes either side uses (``diffusion_model.``, ``model.diffusion_model.``). An fp8-scaled base
(``<module>.weight_scale``, e.g. krea2_raw_fp8_scaled) is dequantized first: the output is a plain file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

from merge_into_model_lokr import kron_delta, read_lokr
from merge_loras_io import read_lora

PREFIXES = ("model.diffusion_model.", "diffusion_model.")
FULL = (".diff", ".diff_b")
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
INTEGER = ("I64", "I32", "I16", "I8", "U8", "BOOL")


def bare(name: str) -> str:
    """A module or key name without its container prefix."""
    return next((name[len(p) :] for p in PREFIXES if name.startswith(p)), name)


@dataclass
class Deltas:
    """Per bare module: LoRA pairs (factor, down, up), LoKr products (factor, w1, w2) and full deltas
    (strength, tensor) for the weight, full deltas for the bias."""

    pairs: dict[str, list[tuple[float, torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    krons: dict[str, list[tuple[float, Any, Any]]] = field(default_factory=dict)
    diffs: dict[str, list[tuple[float, torch.Tensor]]] = field(default_factory=dict)
    bias_diffs: dict[str, list[tuple[float, torch.Tensor]]] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)  # module -> the files that change it

    def modules(self) -> set[str]:
        return set(self.pairs) | set(self.krons) | set(self.diffs) | set(self.bias_diffs)


def read_deltas(inputs: list[tuple[Path, float]]) -> tuple[Deltas, list[dict[str, Any]]]:
    """Every input's deltas, and per input a summary (file, strength, modules, metadata) for the report."""
    deltas, summary = Deltas(), []
    for path, strength in inputs:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            lokr, lokr_keys = read_lokr(handle, list(handle.keys()), strength, path.name)
        # keys the LoKr reader took (their alphas included) are not the LoRA reader's to refuse
        modules, metadata, _ = read_lora(path, skip=FULL + tuple(sorted(lokr_keys)))
        names = set()
        for name, (down, up, alpha) in modules.items():
            deltas.pairs.setdefault(bare(name), []).append((strength * alpha / down.shape[0], down, up))
            names.add(bare(name))
        for name, part in lokr.items():
            deltas.krons.setdefault(bare(name), []).append(part)
            names.add(bare(name))
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in [k for k in handle.keys() if k.endswith(FULL)]:
                bias = key.endswith(".diff_b")
                name = bare(key[: -len(".diff_b" if bias else ".diff")])
                (deltas.bias_diffs if bias else deltas.diffs).setdefault(name, []).append(
                    (strength, handle.get_tensor(key))
                )
                names.add(name)
        if not names:
            raise ValueError(f"{path.name}: no LoRA, LoKr or .diff tensors")
        for name in names:
            deltas.sources.setdefault(name, set()).add(path.name)
        summary.append({"path": path, "strength": strength, "modules": len(names), "metadata": metadata})
    return deltas, summary


def read_float(handle: Any, keys: set[str], key: str) -> torch.Tensor:
    """A floating tensor of a model file in float32, an fp8-scaled weight dequantized with its scale."""
    tensor = handle.get_tensor(key)
    scale = key[: -len(".weight")] + ".weight_scale"
    if tensor.dtype in FP8 and scale in keys:
        return cast(torch.Tensor, tensor.float() * handle.get_tensor(scale).float())
    return cast(torch.Tensor, tensor.float())


def is_plain_float(handle: Any, key: str) -> bool:
    """Whether a model tensor is merged (floating), not copied as stored (integers, flags) or folded (scales)."""
    return not key.endswith(".weight_scale") and handle.get_slice(key).get_dtype() not in INTEGER


def lora_parts(deltas: Deltas, key: str) -> list[tuple[str, Any]]:
    """What adds into this base tensor, each tagged by kind: LoRA pairs, LoKr products and full deltas for a weight,
    ``.diff_b`` for a bias."""
    module, _, kind = bare(key).rpartition(".")
    if kind == "weight":
        return (
            [("pair", p) for p in deltas.pairs.get(module, [])]
            + [("kron", p) for p in deltas.krons.get(module, [])]
            + [("full", p) for p in deltas.diffs.get(module, [])]
        )
    if kind == "bias":
        return [("full", p) for p in deltas.bias_diffs.get(module, [])]
    return []


def add_lora(weight: torch.Tensor, key: str, parts: list[tuple[str, Any]], device: str) -> torch.Tensor:
    w = weight.to(device)
    for kind, part in parts:
        if kind == "pair":  # (factor, down, up)
            delta = part[0] * (part[2].to(device).float() @ part[1].to(device).float())
        elif kind == "kron":  # (factor, w1, w2)
            delta = kron_delta(part, w.shape, key, device)
        else:  # (strength, tensor): a full delta
            delta = part[0] * part[1].to(device).float()
        if delta.shape != w.shape:
            raise ValueError(f"{key}: the delta is {tuple(delta.shape)}, the base weight {tuple(w.shape)}")
        w = w + delta
    return w.cpu()


def check_model(path: Path, handle: Any, keys: set[str]) -> None:
    """Refuse a model file this cannot read exactly: INT8 ConvRot, or a scale on anything but an fp8 weight."""
    if any(k.endswith(".comfy_quant") for k in keys):
        raise ValueError(f"{path.name} is INT8 ConvRot: run scripts/dequantize_int8_convrot.py on it first")
    for scale in (k for k in keys if k.endswith(".weight_scale")):
        stored = handle.get_slice(scale[: -len("_scale")]).get_dtype()
        if stored not in ("F8_E4M3", "F8_E5M2"):
            raise ValueError(f"{path.name}: {scale} scales a {stored} weight; only fp8-scaled models are read")


def check_lora_modules(base: Path, keys: set[str], deltas: Deltas) -> None:
    present = {bare(k[: -len(s)]) for k in keys for s in (".weight", ".bias") if k.endswith(s)}
    missing = sorted(deltas.modules() - present)
    if missing:
        raise ValueError(f"{len(missing)} LoRA modules are not in {base.name}, e.g. {missing[:3]}")
