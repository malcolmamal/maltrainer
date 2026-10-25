#!/usr/bin/env python3
"""Merge checkpoints and LoRAs into one model: a plain single file ComfyUI and the trainers load like the base.

    python scripts/merge_lora_into_model.py base.safetensors out.safetensors a.safetensors:0.8 b.safetensors:1.2
    python scripts/merge_lora_into_model.py turbo.safetensors out.safetensors --add-diff ft.safetensors,raw.safetensors
    python scripts/merge_lora_into_model.py base.safetensors out.safetensors --model other.safetensors:0.3
    python scripts/merge_lora_into_model.py --config templates/examples/merge_into_model_krea2.yaml

Per tensor of the base A, in float32 (merge_into_model_models.py, merge_into_model_core.py):

    W = (1 - sum t) * A + sum t * B                 --model B:t            weighted sum
        + sum s * (B - C)                           --add-diff B,C:s       add difference: C = the model B came from
        + sum strength * (alpha / rank) * up @ down  LoRAs (and LoKr, ComfyUI .diff / .diff_b full deltas)

The YAML layout is in merge_into_model_plan.py and the sample config. Models: bf16 / fp16 / fp32 or fp8-scaled
(dequantized on the way: the output is plain); INT8 ConvRot needs scripts/dequantize_int8_convrot.py first.
Differences between the models (missing or reshaped tensors, a model far from the base, fp8 inputs to a difference)
are printed as warnings and the file is still written; broken LoRAs and unreadable models are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from merge_into_model_core import read_deltas
from merge_into_model_models import merge_into
from merge_into_model_plan import Plan, add_cli, from_yaml
from merge_loras_io import DTYPES, sha256_prefix

SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def base_dtype(base: Path) -> str:
    """The floating dtype most of the base's tensors use; bfloat16 for an fp8 base."""
    counts: dict[str, int] = {}
    with safe_open(str(base), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            name = SAFETENSORS_DTYPES.get(handle.get_slice(key).get_dtype())
            if name:
                counts[name] = counts.get(name, 0) + 1
    return max(counts, key=lambda n: counts[n]) if counts else "bfloat16"


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, nargs="?", help="the base model A (overrides the config's)")
    parser.add_argument("output", type=Path, nargs="?", help="the merged model to write (overrides the config's)")
    parser.add_argument("loras", nargs="*", help="LoRA files, each optionally path:strength (default 1.0)")
    parser.add_argument("--config", type=Path, help="a YAML file with any of the settings")
    parser.add_argument("--model", action="append", metavar="B:T", help="weighted sum with model B at weight T")
    parser.add_argument("--add-diff", action="append", metavar="B,C:S", help="add S x (B - C); C = B's own base")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="output dtype (default: the base's)")
    parser.add_argument("--device", default=None, help="where to merge (default: cuda if available)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser


def _hashes(paths: list[Path]) -> dict[Path, str]:
    """Each input file's SHA-256 prefix, under one progress bar over all their bytes."""
    unique = list(dict.fromkeys(paths))
    total = sum(path.stat().st_size for path in unique)
    print(f"Hashing the {len(unique)} input files for the record", flush=True)
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Hashing") as bar:
        return {path: sha256_prefix(path, on_read=bar.update) for path in unique}


def _recipe(
    plan: Plan, base: Path, dtype_name: str, summary: list[dict[str, object]], hashes: dict[Path, str]
) -> dict[str, object]:
    def entry(path: Path, **extra: object) -> dict[str, object]:
        return {"file": path.name, "sha256": hashes[path], **extra}

    return {
        "base": entry(base),
        "save_dtype": dtype_name,
        "models": [entry(p, weight=t) for p, t in plan.terms.weighted],
        "add_difference": [{"model": entry(b), "base": entry(c), "strength": s} for b, c, s in plan.terms.differences],
        "loras": [entry(Path(str(i["path"])), strength=i["strength"], modules=i["modules"]) for i in summary],
    }


def _plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Plan, Path, Path]:
    """The plan from --config and the command line, checked before anything is read: (plan, base, output)."""
    try:
        plan = add_cli(from_yaml(args.config) if args.config else Plan(), args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if plan.base is None or plan.output is None:
        parser.error("give a base and an output (on the command line or as base: / output: in --config)")
    if not plan.loras and not plan.terms.weighted and not plan.terms.differences:
        parser.error("nothing to merge: give LoRAs, --model or --add-diff (or loras / models / add_difference)")
    for path in plan.paths():
        if not path.is_file():
            parser.error(f"not found: {path}")
    if plan.output.exists() and not args.force:
        parser.error(f"{plan.output} exists; pass --force to overwrite")
    return plan, plan.base, plan.output


def _announce(plan: Plan, base: Path) -> None:
    """What goes into the merge, before the LoRAs are read."""
    share = 1.0 - sum(t for _, t in plan.terms.weighted)
    print(f"  base {base.name}" + (f" at weight {share:g}" if plan.terms.weighted else ""))
    for path, weight in plan.terms.weighted:
        print(f"  model {path.name} at weight {weight:g}")
    for model, origin, strength in plan.terms.differences:
        print(f"  difference {model.name} - {origin.name} at strength {strength:g}")


def main() -> int:
    parser = _arguments()
    args = parser.parse_args()
    plan, base, output = _plan(parser, args)
    dtype_name = plan.dtype or base_dtype(base)
    device = plan.device or ("cuda" if torch.cuda.is_available() else "cpu")

    started = time.time()
    try:
        _announce(plan, base)
        deltas, summary = read_deltas(plan.loras)
        for item in summary:
            print(f"  LoRA {item['path'].name}: {item['modules']} modules at strength {item['strength']}")
        out, metadata, report = merge_into(base, plan.terms, deltas, DTYPES[dtype_name], device)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    recipe = _recipe(plan, base, dtype_name, summary, _hashes([base, *plan.paths()]))
    print(f"Writing {output.name}...", flush=True)
    save_file(out, str(tmp), metadata=dict(metadata, maltrainer_merge_into_model=json.dumps(recipe)))
    tmp.replace(output)
    for label, value in report["distances"].items():
        print(f"  distance {label}: {value:.2%}")
    print(
        f"wrote {output} ({output.stat().st_size / 1e9:.2f} GB, {dtype_name}): "
        f"{report['merged_tensors']} tensors changed, by {report['mean_change']:.2%} on average "
        f"(at most {report['max_change']:.2%}) in {time.time() - started:.0f}s"
    )
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
