"""What scripts/merge_lora_into_model.py merges: from a YAML file, the command line, or both.

    base: C:/Development/ComfyUI/models/diffusion_models/Krea2/krea2_turbo_fp8_scaled.safetensors
    output: output/krea2_turbo_people_bf16.safetensors
    dtype: bfloat16            # optional; default: the base's own (bf16 for an fp8 base)
    device: cuda               # optional; default: cuda if available
    models:                    # optional, weighted sum: (1 - sum of weights) * base + weight * model
      - path: Krea2Realism_V1_fp16.safetensors
        weight: 0.3
    add_difference:            # optional: + strength * (model - its base), the base it was trained from
      - model: output/krea2_finetune/krea/krea2_ft_krea.safetensors
        base: output/krea2_finetune/krea/_base_dequantized_bf16.safetensors
        strength: 1.0          # optional, default 1.0
    loras:                     # optional; a bare path means strength 1.0
      - path: Krea2_TextFusion_Refusal_Reduction.safetensors
        strength: 1.0
      - another_lora.safetensors

Relative paths are from the current directory, as in the training configs. The command line's base, output,
--dtype and --device override the file's; its --model, --add-diff and LoRAs are added to the file's lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merge_into_model_models import Terms
from merge_loras_io import DTYPES, parse_input

TOP = {"base", "output", "dtype", "device", "models", "add_difference", "loras"}


@dataclass
class Plan:
    base: Path | None = None
    output: Path | None = None
    dtype: str | None = None
    device: str | None = None
    terms: Terms = field(default_factory=Terms)
    loras: list[tuple[Path, float]] = field(default_factory=list)

    def paths(self) -> list[Path]:
        """Every input file, for the existence check."""
        found = [p for p, _ in self.terms.weighted] + [p for b, c, _ in self.terms.differences for p in (b, c)]
        return ([self.base] if self.base else []) + found + [p for p, _ in self.loras]


def _entries(config: dict[str, Any], name: str, source: Path) -> list[Any]:
    value = config.get(name) or []
    if not isinstance(value, list):
        raise ValueError(f"{source}: '{name}' must be a list")
    return value


def _entry(raw: Any, allowed: set[str], required: set[str], where: str) -> dict[str, Any]:
    entry = {"path": raw} if isinstance(raw, str) else raw
    if not isinstance(entry, dict) or not required <= set(entry):
        raise ValueError(f"{where} needs {', '.join(sorted(required))}")
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    return entry


def from_yaml(path: Path) -> Plan:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle.read().replace("\xa0", " ")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping of settings")
    unknown = set(config) - TOP
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (known: {sorted(TOP)})")
    if config.get("dtype") is not None and config["dtype"] not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {sorted(DTYPES)}, got {config['dtype']!r}")
    plan = Plan(
        base=Path(config["base"]) if config.get("base") else None,
        output=Path(config["output"]) if config.get("output") else None,
        dtype=config.get("dtype"),
        device=config.get("device"),
    )
    for i, raw in enumerate(_entries(config, "models", path)):
        e = _entry(raw, {"path", "weight"}, {"path", "weight"}, f"{path}: models[{i}]")
        plan.terms.weighted.append((Path(e["path"]), float(e["weight"])))
    for i, raw in enumerate(_entries(config, "add_difference", path)):
        e = _entry(raw, {"model", "base", "strength"}, {"model", "base"}, f"{path}: add_difference[{i}]")
        plan.terms.differences.append((Path(e["model"]), Path(e["base"]), float(e.get("strength", 1.0))))
    for i, raw in enumerate(_entries(config, "loras", path)):
        e = _entry(raw, {"path", "strength"}, {"path"}, f"{path}: loras[{i}]")
        plan.loras.append((Path(e["path"]), float(e.get("strength", 1.0))))
    return plan


def add_cli(plan: Plan, args: Any) -> Plan:
    """The command line on top of the file's plan (or an empty one)."""
    plan.base = args.base or plan.base
    plan.output = args.output or plan.output
    plan.dtype = args.dtype or plan.dtype
    plan.device = args.device or plan.device
    plan.terms.weighted += [parse_input(spec) for spec in args.model or []]
    for spec in args.add_diff or []:
        pair, strength = parse_input(spec)
        model, comma, base = str(pair).partition(",")
        if not comma or not model or not base:
            raise ValueError(f"--add-diff {spec!r}: give MODEL,BASE or MODEL,BASE:STRENGTH")
        plan.terms.differences.append((Path(model), Path(base), strength))
    plan.loras += [parse_input(spec) for spec in args.loras or []]
    return plan
