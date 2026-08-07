#!/usr/bin/env python3
"""Inspect the training metadata embedded in a maltrainer LoRA safetensors file.

Reads only the file's metadata header, so it is instant even on a multi-gigabyte
checkpoint and never allocates the tensors.

    # human-readable summary
    python scripts/inspect_lora_metadata.py output/minimax_h3_lora/minimax_h3_lora_step_2250.safetensors

    # every metadata key, verbatim
    python scripts/inspect_lora_metadata.py MODEL.safetensors --all

    # recover the training config as runnable YAML
    python scripts/inspect_lora_metadata.py MODEL.safetensors --config -o recovered.yaml

    # raw JSON, for scripting
    python scripts/inspect_lora_metadata.py MODEL.safetensors --json

    # what changed between two runs
    python scripts/inspect_lora_metadata.py A.safetensors --diff B.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maltrainer.utils.lora_metadata import CONFIG_KEY, extract_config, read_metadata  # noqa: E402

# Keys worth showing first in the summary, in reading order.
SUMMARY_ORDER = (
    ("step", "step"),
    ("maltrainer_planned_total_steps", "planned total steps"),
    ("maltrainer_dataset_samples", "dataset samples"),
    ("maltrainer_steps_per_epoch", "steps/epoch"),
    ("ss_num_epochs", "epochs"),
    ("ss_resolution", "resolution"),
    ("ss_network_dim", "rank"),
    ("ss_network_alpha", "alpha"),
    ("maltrainer_lora_layers", "lora layers"),
    ("maltrainer_trainable_parameters", "trainable params"),
    ("maltrainer_lora_dtype", "adapter dtype"),
    ("maltrainer_save_dtype", "saved dtype"),
    ("ss_optimizer", "optimizer"),
    ("maltrainer_optimizer_resolved", "optimizer args"),
    ("ss_learning_rate", "learning rate"),
    ("ss_weight_decay", "weight decay"),
    ("ss_lr_warmup_steps", "warmup steps"),
    ("ss_batch_size", "batch size"),
    ("ss_gradient_accumulation_steps", "grad accum"),
    ("ss_max_grad_norm", "max grad norm"),
    ("ss_timestep_sampling", "timestep sampling"),
    ("maltrainer_resolved_sigma_shift", "resolved sigma shift"),
    ("ss_sigmoid_scale", "sigmoid scale"),
    ("ss_discrete_flow_shift", "timestep_shift (if fixed)"),
    ("maltrainer_latent_size", "latent size"),
    ("maltrainer_int8_grad_precision", "int8 backward"),
    ("ss_mixed_precision", "mixed precision"),
    ("ss_seed", "seed"),
    ("ss_transformer", "transformer"),
    ("ss_vae", "vae"),
    ("ss_text_encoder", "text encoder"),
    ("maltrainer_git_commit", "trainer commit"),
    ("maltrainer_torch_version", "torch"),
    ("maltrainer_started_at", "run started"),
    ("maltrainer_saved_at", "checkpoint saved"),
)


def _fmt(value: str) -> str:
    """Pretty-print a value that may be a JSON blob."""
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.dumps(json.loads(stripped), sort_keys=True)
        except json.JSONDecodeError:
            pass
    if value.isdigit() and len(value) > 6:
        return f"{int(value):,}"
    return value


def print_summary(path: Path, metadata: Dict[str, str]) -> None:
    print(f"{path.name}")
    size_gb = path.stat().st_size / 1e9
    print(f"  file size: {size_gb:.2f} GB, {len(metadata)} metadata keys")
    if not metadata:
        print("  (no metadata -- not written by maltrainer, or written before embedding existed)")
        return
    print()
    width = max(len(label) for _, label in SUMMARY_ORDER)
    for key, label in SUMMARY_ORDER:
        if key in metadata and metadata[key] != "":
            print(f"  {label:<{width}}  {_fmt(metadata[key])}")

    shown = {key for key, _ in SUMMARY_ORDER} | {CONFIG_KEY}
    extra = sorted(k for k in metadata if k not in shown)
    if extra:
        print(f"\n  {len(extra)} further key(s): {', '.join(extra)}")
    print(f"\n  embedded config: {'yes' if CONFIG_KEY in metadata else 'no'} (--config to recover)")


def dump_config(metadata: Dict[str, str], out: Optional[Path]) -> int:
    config: Any = extract_config(metadata)
    if config is None:
        print(f"No embedded config (looked for '{CONFIG_KEY}').", file=sys.stderr)
        return 1
    try:
        import yaml

        text = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(config, indent=2)
        print("PyYAML unavailable; emitting JSON instead.", file=sys.stderr)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text, end="")
    return 0


def diff(left: Dict[str, str], right: Dict[str, str], left_name: str, right_name: str) -> None:
    # The config blob restates everything else, and saved_at always differs, so comparing
    # them adds noise rather than information.
    ignore = {CONFIG_KEY, "maltrainer_saved_at", "maltrainer_started_at", "step"}
    keys = sorted((set(left) | set(right)) - ignore)
    rows = [(k, left.get(k, "--"), right.get(k, "--")) for k in keys if left.get(k) != right.get(k)]
    print(f"< {left_name}\n> {right_name}\n")
    if not rows:
        print("Identical (ignoring step and timestamps).")
        return
    width = max(len(k) for k, _, _ in rows)
    for key, lhs, rhs in rows:
        print(f"  {key:<{width}}\n      < {_fmt(lhs)}\n      > {_fmt(rhs)}")
    print(f"\n{len(rows)} difference(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="path to a .safetensors LoRA")
    parser.add_argument("--all", action="store_true", help="print every metadata key verbatim")
    parser.add_argument("--json", action="store_true", help="print all metadata as JSON")
    parser.add_argument("--config", action="store_true", help="recover the embedded training config")
    parser.add_argument("-o", "--output", type=Path, help="write --config to this file instead of stdout")
    parser.add_argument("--diff", type=Path, metavar="OTHER", help="compare against another checkpoint")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"No such file: {args.model}", file=sys.stderr)
        return 1

    try:
        metadata = read_metadata(args.model)
    except Exception as error:  # safetensors raises a variety of types on malformed input
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
