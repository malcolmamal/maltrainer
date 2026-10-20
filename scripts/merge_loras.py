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
import sys
from pathlib import Path
from typing import Any

import yaml
from safetensors.torch import save_file

from merge_loras_core import merge, merged_metadata
from merge_loras_io import (  # noqa: F401  (re-exported)
    DTYPES,
    LAYOUTS,
    load_config,
    parse_input,
    read_lora,
    sha256_prefix,
)


def _plan(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[list[tuple[Path, float]], Path, str, bool]:
    """The inputs, output, saved dtype name and verify flag, from the config file or the command line."""
    if args.config:
        if args.inputs:
            parser.error("give the LoRAs either in --config or on the command line, not both")
        try:
            config = load_config(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
        output = args.output or (Path(config["output"]) if config["output"] else None)
        if output is None:
            parser.error("no output: set 'output:' in the config or pass it on the command line")
        return config["inputs"], output, args.dtype or config["dtype"] or "float32", args.verify or config["verify"]
    if args.output is None or not args.inputs:
        parser.error("give an output and at least one LoRA, or --config")
    return [parse_input(spec) for spec in args.inputs], args.output, args.dtype or "float32", args.verify


def _report(output: Path, info: dict[str, Any], dtype_name: str, verify: bool) -> None:
    size = output.stat().st_size / 1024**2
    print(f"Wrote {output} ({size:.0f} MB): {info['modules']} modules at rank {info['ranks']}, {dtype_name}")
    if info["partial_modules"]:
        print(f"  {info['partial_modules']} modules came from only some of the inputs")
    if verify:
        print(f"  verify: worst module differs from the stacked inputs by {info['max_rel_error']:.2e} (relative)")
    print("Load it at strength 1.0: the strengths above are already folded in.")


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

    inputs, output, dtype_name, verify = _plan(parser, args)
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
    _report(output, info, dtype_name, verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
