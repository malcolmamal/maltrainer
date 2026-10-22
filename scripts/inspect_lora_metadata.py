#!/usr/bin/env python3
"""Inspect the training metadata embedded in a maltrainer LoRA safetensors file.

Reads only the file's metadata header, so it is instant even on a multi-gigabyte
checkpoint and never allocates the tensors.

Usage:

  Summary of how a checkpoint was trained (default):
    python scripts/inspect_lora_metadata.py output/minimax_h3_lora/..._step_2250.safetensors

  Recover the training config as runnable YAML, to reproduce or fork a run:
    python scripts/inspect_lora_metadata.py MODEL.safetensors --config
    python scripts/inspect_lora_metadata.py MODEL.safetensors --config -o recovered.yaml
    python train_minimax_h3_lora.py --config recovered.yaml

  Compare two checkpoints -- what ACTUALLY differed between two runs:
    python scripts/inspect_lora_metadata.py A.safetensors --diff B.safetensors

  Every key, for grepping or scripting:
    python scripts/inspect_lora_metadata.py MODEL.safetensors --all
    python scripts/inspect_lora_metadata.py MODEL.safetensors --json

Key naming: ss_* are flat scalars in the kohya/musubi convention; maltrainer_* are this
trainer's own record, including values resolved at runtime that the config cannot state
(the sigma shift chosen for the latent size, the dtype adapters trained in, the optimizer
arguments the installed library accepted). See docs/lora-metadata.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_lora_render import (  # noqa: F401  (names the tests and callers import from here)
    SUMMARY_ORDER,
    _abbreviate,
    _fmt,
    diff,
    dump_config,
    print_summary,
)
from maltrainer.utils.lora_metadata import read_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "  * Checkpoints written before metadata embedding existed report 'no metadata'\n"
            "    rather than failing. Only runs started after that change carry it.\n"
            "  * The embedded config contains absolute local paths (dataset, base models).\n"
            "    Set output.embed_config: false in the training config to omit it.\n"
            "  * --diff ignores the config blob and timestamps, so it reports only settings\n"
            "    that genuinely differ.\n"
        ),
    )
    parser.add_argument("model", type=Path, help="path to a .safetensors LoRA checkpoint")
    parser.add_argument("--all", action="store_true", help="print every metadata key as key=value, sorted")
    parser.add_argument("--json", action="store_true", help="print all metadata as a JSON object")
    parser.add_argument(
        "--config",
        action="store_true",
        help="recover the embedded training config as YAML (runnable with --config)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="with --config, write the YAML to FILE instead of stdout",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        metavar="OTHER",
        help="compare this checkpoint's settings against OTHER and list the differences",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"No such file: {args.model}", file=sys.stderr)
        return 1

    try:
        metadata = read_metadata(args.model)
    except (OSError, ValueError) as error:  # malformed input
        print(f"Could not read metadata from {args.model}: {error}", file=sys.stderr)
        return 1

    if args.diff:
        if not args.diff.exists():
            print(f"No such file: {args.diff}", file=sys.stderr)
            return 1
        diff(metadata, read_metadata(args.diff), args.model.name, args.diff.name)
        return 0
    if args.config:
        return dump_config(metadata, args.output)
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    if args.all:
        for key in sorted(metadata):
            print(f"{key}={metadata[key]}")
        return 0

    print_summary(args.model, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
