#!/usr/bin/env python3
"""Merge LoRAs into one file that applies exactly what loading them together does.

Stacking LoRAs in ComfyUI adds their weight deltas. For one module, LoRA i contributes

    delta_i = strength_i * (alpha_i / rank_i) * B_i @ A_i

and the sum of those is itself a LoRA of rank sum(rank_i). Stack the down matrices, set the up
matrices side by side with each LoRA's factor folded in, and set alpha = total rank (scale 1):

    A = [A_1; A_2; ...]            (sum rank) x in
    B = [c_1 B_1, c_2 B_2, ...]    out x (sum rank),   c_i = strength_i * alpha_i / rank_i
    B @ A = sum_i c_i B_i @ A_i

Nothing is approximated: no SVD, no rank truncation. Saved in float32 (the default), the only
difference from the stack is float32 rounding of c_i * B_i, ~1e-7 relative. ``--verify`` measures
it per module against the fp64 sum.

    python scripts/merge_loras.py out.safetensors a.safetensors b.safetensors
    python scripts/merge_loras.py out.safetensors a.safetensors:0.8 b.safetensors:1.2 --verify
    python scripts/merge_loras.py out.safetensors a.safetensors:1.5     (one LoRA: bake a strength in)
    python scripts/merge_loras.py --config templates/examples/merge_moss_image_feliciaday_soup.yaml

A YAML config lists the LoRAs and their strengths (relative paths are from the current directory,
as in the training configs):

    output: output/soup.safetensors
    dtype: float32          # optional, default float32
    verify: true            # optional
    loras:
      - path: output/run_a/a_step_500.safetensors
        strength: 1.0       # optional, default 1.0
      - output/run_b/b_step_750.safetensors     # a bare path means strength 1.0

Command-line ``output``, ``--dtype`` and ``--verify`` override the config; overwriting an
existing output always needs ``--force``.

Key layouts: ``<module>.lora_A.weight`` / ``lora_B.weight`` (maltrainer, PEFT) or
``<module>.lora_down.weight`` / ``lora_up.weight`` (kohya), each with an optional
``<module>.alpha`` (missing = rank, i.e. scale 1). A module present in only some inputs is merged
from those. Other tensors (DoRA scales, LoKr, ...) are refused rather than silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

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


def read_lora(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict[str, str], str]:
    """module -> (down [r, in], up [out, r], alpha); the file's metadata; the down/up layout."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in keys}

    layout = next((pair for pair in LAYOUTS if any(k.endswith(pair[0]) for k in keys)), None)
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


def sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def merge(
    inputs: list[tuple[Path, float]], dtype: torch.dtype, verify: bool
) -> tuple[dict[str, torch.Tensor], dict, list]:
    loras: list[dict] = []
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


def load_config(path: Path) -> dict:
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


def merged_metadata(loras: list, info: dict, dtype: torch.dtype, config_file: Path | None = None) -> dict[str, str]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, nargs="?", help="merged .safetensors to write (overrides the config's)")
    parser.add_argument("inputs", nargs="*", help="LoRA files, each optionally path:strength (default 1.0)")
    parser.add_argument("--config", type=Path, help="YAML file listing the LoRAs and their strengths")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default=None,
        help="saved dtype. float32 (default) is exact; bfloat16 halves the size at ~2e-3 relative rounding",
    )
    parser.add_argument("--verify", action="store_true", help="check every module against the fp64 sum of the inputs")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args()

    if args.config:
        if args.inputs:
            parser.error("give the LoRAs either in --config or on the command line, not both")
        try:
            config = load_config(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
        inputs = config["inputs"]
        output = args.output or (Path(config["output"]) if config["output"] else None)
        dtype_name = args.dtype or config["dtype"] or "float32"
        verify = args.verify or config["verify"]
        if output is None:
            parser.error("no output: set 'output:' in the config or pass it on the command line")
    else:
        if args.output is None or not args.inputs:
            parser.error("give an output and at least one LoRA, or --config")
        inputs = [parse_input(spec) for spec in args.inputs]
        output, dtype_name, verify = args.output, args.dtype or "float32", args.verify

    for path, _ in inputs:
        if not path.is_file():
            parser.error(f"not found: {path}")
    if output.exists() and not args.force:
        parser.error(f"{output} exists; pass --force to overwrite")
    dtype = DTYPES[dtype_name]

    print(
        f"Merging {len(inputs)} LoRA{'s' if len(inputs) > 1 else ''}"
        + (f" from {args.config}" if args.config else "")
        + ":"
    )
    try:
        state, info, loras = merge(inputs, dtype, verify)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    bases = {lora["metadata"].get("ss_base_model_version") for lora in loras} - {None}
    if len(bases) > 1:
        print(f"warning: the inputs were trained on different base models: {sorted(bases)}", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata=merged_metadata(loras, info, dtype, args.config))
    size = output.stat().st_size / 1024**2
    print(f"Wrote {output} ({size:.0f} MB): {info['modules']} modules at rank {info['ranks']}, {dtype_name}")
    if info["partial_modules"]:
        print(f"  {info['partial_modules']} modules came from only some of the inputs")
    if verify:
        print(f"  verify: worst module differs from the stacked inputs by {info['max_rel_error']:.2e} (relative)")
    print("Load it at strength 1.0: the strengths above are already folded in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
