"""The merge itself, for scripts/merge_loras.py: exact rank concatenation and the metadata it records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from merge_loras_io import read_lora, sha256_prefix


def merge(
    inputs: list[tuple[Path, float]], dtype: torch.dtype, verify: bool
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[Any]]:
    loras: list[dict[str, Any]] = []
    for path, strength in inputs:
        modules, metadata, layout = read_lora(path)
        ranks = sorted({down.shape[0] for down, _, _ in modules.values()})
        alphas = sorted({alpha for _, _, alpha in modules.values()})
        loras.append({"path": path, "strength": strength, "modules": modules, "metadata": metadata, "layout": layout})
        print(f"  {path.name}: {len(modules)} modules, {layout}, rank {ranks}, alpha {alphas}, strength {strength}")

    out_down, out_up = loras[0]["layout"].split("/")
    names = sorted(set().union(*(lora["modules"] for lora in loras)))
    state: dict[str, torch.Tensor] = {}
    worst = 0.0
    partial = 0
    for name in names:
        parts = []
        for lora in loras:
            if name not in lora["modules"]:
                continue
            down, up, alpha = lora["modules"][name]
            factor = lora["strength"] * alpha / down.shape[0]
            parts.append((down.double(), up.double(), factor))
        if len(parts) < len(loras):
            partial += 1
        shapes = {(up.shape[0], down.shape[1]) for down, up, _ in parts}
        if len(shapes) > 1:
            raise ValueError(
                f"{name}: inputs disagree on the layer shape {sorted(shapes)}; are they for the same model?"
            )

        down = torch.cat([d for d, _, _ in parts], dim=0)
        up = torch.cat([u * f for _, u, f in parts], dim=1)
        total_rank = down.shape[0]
        state[f"{name}.{out_down}.weight"] = down.to(dtype).contiguous()
        state[f"{name}.{out_up}.weight"] = up.to(dtype).contiguous()
        state[f"{name}.alpha"] = torch.tensor(float(total_rank), dtype=torch.float32)
        if verify:
            expected = sum(f * (u @ d) for d, u, f in parts)
            merged = state[f"{name}.{out_up}.weight"].double() @ state[f"{name}.{out_down}.weight"].double()
            err = ((merged - expected).norm() / expected.norm().clamp_min(1e-30)).item()
            worst = max(worst, err)

    info = {
        "modules": len(names),
        "partial_modules": partial,
        "max_rel_error": worst if verify else None,
        "ranks": sorted({state[k].shape[0] for k in state if k.endswith(f".{out_down}.weight")}),
    }
    return state, info, loras


def merged_metadata(
    loras: list[Any], info: dict[str, Any], dtype: torch.dtype, config_file: Path | None = None
) -> dict[str, str]:
    """Carry the inputs' shared ``ss_*`` identity, fix the network size, and record the recipe."""
    out: dict[str, str] = {}
    first = loras[0]["metadata"]
    for key in ("ss_base_model_version", "ss_network_module", "modelspec.architecture"):
        values = {lora["metadata"].get(key) for lora in loras}
        if len(values) == 1 and first.get(key) is not None:
            out[key] = first[key]
    rank = info["ranks"][0] if len(info["ranks"]) == 1 else max(info["ranks"])
    out["ss_network_dim"] = str(rank)
    out["ss_network_alpha"] = str(rank)
    out["maltrainer_merge"] = json.dumps(
        {
            "method": "exact rank concatenation (B @ A = sum of strength * alpha / rank * B_i @ A_i)",
            "save_dtype": str(dtype).replace("torch.", ""),
            "max_rel_error": info["max_rel_error"],
            "config_file": config_file.name if config_file else None,
            "inputs": [
                {
                    "file": lora["path"].name,
                    "sha256": sha256_prefix(lora["path"]),
                    "strength": lora["strength"],
                    "base_model": lora["metadata"].get("ss_base_model_version"),
                    "optimizer": lora["metadata"].get("ss_optimizer"),
                    # each input's full training record, if it had one
                    "maltrainer_config": lora["metadata"].get("maltrainer_config"),
                }
                for lora in loras
            ],
        }
    )
    return out
