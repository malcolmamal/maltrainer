#!/usr/bin/env python3
"""Extract a LoRA from a full finetune: the SVD of each layer's change against the base model.

    python scripts/extract_lora.py base.safetensors finetuned.safetensors                  -> rank 64
    python scripts/extract_lora.py base.safetensors ft.safetensors --rank 16 32 64         -> three files, one SVD pass
    python scripts/extract_lora.py base.safetensors ft.safetensors --mode fro --value 0.9 --rank 128

Writes ``<finetuned>_lora_r<rank>.safetensors`` next to the finetune (or into ``--output-dir``).
``--mode`` picks each layer's rank (see maltrainer/finetune/extract.py): ``fixed`` = the rank,
``fro`` = enough to hold --value of the layer's change, ``ratio`` = singular values >= --value x
the largest, ``threshold`` = singular values >= --value; the dynamic modes stay under --rank.

The model is detected from the checkpoint (MOSS-Image: fused q/k/v split into the q / k / v LoRA
modules ComfyUI expects); anything else extracts every changed 2-D weight as is. Neither input is
modified. Both must be plain bf16/fp16/fp32 files with the same tensors (a finetune saved by this
repo and the base it started from).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maltrainer.finetune.extract import GENERIC, MODES, extract, strip_prefix  # noqa: E402
from maltrainer.models.moss_image.finetune import EXTRACT_PROFILE as MOSS_IMAGE  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
PROFILES = {"moss_image": MOSS_IMAGE, "generic": GENERIC}


def detect_profile(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    prefix = strip_prefix(keys)
    names = [k[len(prefix) :] for k in keys]
    for profile in PROFILES.values():
        if profile.detect(names):
            return profile
    return GENERIC


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="the base checkpoint the finetune started from")
    parser.add_argument("finetuned", type=Path, help="the finetuned checkpoint")
    parser.add_argument(
        "--rank", type=int, nargs="+", default=[64], help="rank (cap) per layer; several = several files (default 64)"
    )
    parser.add_argument(
        "--mode", choices=MODES, default="fixed", help="how each layer's rank is chosen (default fixed)"
    )
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help="fro: share of the change to keep (e.g. 0.9); ratio: share of the largest singular value; threshold: absolute",
    )
    parser.add_argument(
        "--model", choices=["auto", *PROFILES], default="auto", help="checkpoint layout (default: detected)"
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="saved dtype (default bfloat16)")
    parser.add_argument("--device", default=None, help="cuda or cpu for the SVDs (default: cuda if available)")
    parser.add_argument("--output-dir", type=Path, help="where to write (default: next to the finetune)")
    parser.add_argument("--name", help="output name stem (default: <finetuned>_lora)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    for path in (args.base, args.finetuned):
        if not path.is_file():
            parser.error(f"not found: {path}")
    if any(r < 1 for r in args.rank):
        parser.error("--rank must be at least 1")
    if args.mode != "fixed" and args.value is None:
        parser.error(f"--mode {args.mode} needs --value")
    if args.mode == "fro" and not 0.0 < args.value <= 1.0:
        parser.error("--mode fro needs --value in (0, 1]")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    profile = PROFILES[args.model] if args.model != "auto" else detect_profile(args.base)

    out_dir = args.output_dir or args.finetuned.parent
    stem = args.name or f"{args.finetuned.stem}_lora"
    targets = {rank: out_dir / f"{stem}_r{rank}.safetensors" for rank in sorted(set(args.rank))}
    for target in targets.values():
        if target.exists() and not args.force:
            parser.error(f"{target} exists; pass --force to overwrite")

    print(
        f"{args.finetuned.name} - {args.base.name}: {profile.name} layout, mode {args.mode}"
        f"{f' {args.value:g}' if args.value is not None else ''}, rank {sorted(targets)} on {device}"
    )
    started = time.time()
    try:
        states, report = extract(
            args.base,
            args.finetuned,
            list(targets),
            mode=args.mode,
            value=args.value or 0.0,
            profile=profile,
            device=device,
            dtype=DTYPES[args.dtype],
            progress=lambda names: tqdm(names, desc="SVD per layer"),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"  {report['unchanged_tensors']} tensors unchanged")
    if report["changed_but_not_extracted"]:
        skipped = report["changed_but_not_extracted"]
        print(f"  {len(skipped)} changed tensors are not LoRA targets and are left out, e.g. {skipped[:3]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, target in targets.items():
        info = report["per_cap"][rank]
        metadata = {
            "ss_network_dim": str(info["rank_max"]),
            "ss_network_alpha": str(info["rank_max"]),
            "maltrainer_architecture": f"{profile.name}_extracted",
            "maltrainer_extract": json.dumps(
                {
                    "method": "SVD per layer of finetuned - base",
                    "mode": args.mode,
                    "value": args.value,
                    "rank_cap": rank,
                    "profile": profile.name,
                    "save_dtype": args.dtype,
                    "base": {"file": args.base.name, "bytes": args.base.stat().st_size},
                    "finetuned": {"file": args.finetuned.name, "bytes": args.finetuned.stat().st_size},
                }
            ),
        }
        save_file(states[rank], str(target), metadata=metadata)
        print(
            f"  r{rank}: {info['modules']} modules, rank {info['rank_min']}..{info['rank_max']} (mean {info['rank_mean']:.1f}), "
            f"dropped norm per layer: mean {info['dropped_mean']:.1%}, worst {info['dropped_worst']:.1%}"
        )
        print(f"       wrote {target} ({target.stat().st_size / 1024**2:.0f} MB)")
    print(f"  done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
