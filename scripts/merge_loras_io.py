"""Reading side of scripts/merge_loras.py: LoRA files, ``path:strength`` specs, the YAML config and file hashes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

LAYOUTS = ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"))
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_input(spec: str) -> tuple[Path, float]:
    """``path`` or ``path:strength``. A Windows drive colon (C:\\...) is not a strength separator."""
    head, sep, tail = spec.rpartition(":")
    if sep and head and not head.endswith(("\\", "/")) and len(head) > 1:
        try:
            return Path(head), float(tail)
        except ValueError:
            pass
    return Path(spec), 1.0


def read_lora(
    path: Path, skip: tuple[str, ...] = ()
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict[str, str], str]:
    """module -> (down [r, in], up [out, r], alpha); the file's metadata; the down/up layout ("" for none).

    ``skip``: key suffixes another reader handles (merge_lora_into_model.py's full ``.diff`` deltas); a file of
    only those has no layout and no modules. Any other tensor this does not understand is refused."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if not key.endswith(skip)] if skip else list(handle.keys())
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in keys}
    layout = next((pair for pair in LAYOUTS if any(k.endswith(pair[0]) for k in keys)), None)
    if layout is None and skip and not keys:
        return {}, metadata, ""
    if layout is None:
        raise ValueError(f"{path.name}: no lora_A/lora_B or lora_down/lora_up keys")
    down_suffix, up_suffix = layout
    modules, used = {}, set()
    for key in keys:
        if not key.endswith(down_suffix):
            continue
        base = key[: -len(down_suffix)]
        up_key, alpha_key = base + up_suffix, base + ".alpha"
        if up_key not in tensors:
            raise ValueError(f"{path.name}: {key} has no matching {up_key}")
        down, up = tensors[key], tensors[up_key]
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(
                f"{path.name}: {base} is not a plain linear LoRA (down {tuple(down.shape)}, up {tuple(up.shape)})"
            )
        rank = down.shape[0]
        alpha = float(tensors[alpha_key].item()) if alpha_key in tensors else float(rank)
        modules[base] = (down, up, alpha)
        used.update({key, up_key, alpha_key})

    leftover = sorted(set(keys) - used)
    if leftover:
        raise ValueError(
            f"{path.name}: tensors this merge does not understand, e.g. {leftover[:3]} ({len(leftover)} total)"
        )
    return modules, metadata, "/".join(s.split(".")[1] for s in layout)


def sha256_prefix(path: Path, length: int = 12, on_read: Callable[[int], object] | None = None) -> str:
    """The first ``length`` hex digits of the file's SHA-256; ``on_read(bytes)`` after each block (a progress bar)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            if on_read is not None:
                on_read(len(block))
    return digest.hexdigest()[:length]


def load_config(path: Path) -> dict[str, Any]:
    """The YAML merge config -> {"output", "dtype", "verify", "inputs": [(Path, strength)]}."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = handle.read().replace("\xa0", " ")
        config = yaml.safe_load(raw) or {}
    if not isinstance(config, dict) or not isinstance(config.get("loras"), list) or not config["loras"]:
        raise ValueError(f"{path}: needs a non-empty 'loras:' list")
    inputs = []
    for i, raw in enumerate(config["loras"]):
        entry = {"path": raw} if isinstance(raw, str) else raw
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"{path}: loras[{i}] needs a 'path'")
        unknown = set(entry) - {"path", "strength"}
        if unknown:
            raise ValueError(f"{path}: loras[{i}] has unknown keys {sorted(unknown)}")
        inputs.append((Path(entry["path"]), float(entry.get("strength", 1.0))))
    unknown = set(config) - {"output", "dtype", "verify", "loras"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    dtype = config.get("dtype")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    return {
        "output": config.get("output"),
        "dtype": dtype,
        "verify": bool(config.get("verify", False)),
        "inputs": inputs,
    }
