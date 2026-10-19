#!/usr/bin/env python3
"""Shrink a LoRA to a lower rank (e.g. rank 32 -> 8) with an SVD of each layer's update.

For one module the LoRA applies

    W = (alpha / rank) * B @ A            B: out x rank, A: rank x in

This keeps the fewest singular values of W that hold ``--frac`` of its Frobenius norm (the same
rule as kohya's ``resize_lora.py --dynamic_method sv_fro``), at most ``--rank`` of them, and writes
them back as a smaller lora_A / lora_B pair with alpha = the new rank (scale 1).

The SVD runs on a small rank x rank core, not on the full out x in matrix. QR-factor B and A^T,
then W = Qb @ (Rb @ Ra^T) @ Qa^T, so the singular values of W are those of the tiny core. That is
exact, and takes seconds instead of minutes. It also gives the error for free: the share of W's
norm that was dropped is sqrt(sum of the dropped s^2 / sum of all s^2), printed per file.

    python scripts/resize_lora.py a.safetensors                    -> a_resized.safetensors, rank 8
    python scripts/resize_lora.py a.safetensors b.safetensors --rank 16
    python scripts/resize_lora.py a.safetensors --output out/a_r8.safetensors

The input is never modified. Key layouts and refusals are the same as ``merge_loras.py``
(lora_A/lora_B or lora_down/lora_up, plain linear layers only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file

from merge_loras import DTYPES, read_lora, sha256_prefix


def resize_module(
    down: torch.Tensor, up: torch.Tensor, alpha: float, max_rank: int, frac: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """(new_down [k, in], new_up [out, k], k, relative_error) for one module, in float32."""
    factor = alpha / down.shape[0]
    q_up, r_up = torch.linalg.qr(up.to(device, torch.float32))
    q_down, r_down = torch.linalg.qr(down.to(device, torch.float32).T)
    core = factor * (r_up @ r_down.T)
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    energy = s.double() ** 2
    total = energy.sum().item()
    if total <= 0.0:
        keep, error = 1, 0.0
    else:
        cumulative = (energy.cumsum(0) / total).tolist()
        keep = next((i + 1 for i, share in enumerate(cumulative) if share >= frac), len(cumulative))
        keep = max(1, min(keep, max_rank, len(s)))
        error = (energy[keep:].sum().item() / total) ** 0.5

    root = s[:keep].sqrt()
    new_up = (q_up @ u[:, :keep]) * root[None, :]
    new_down = root[:, None] * (vh[:keep] @ q_down.T)
    return new_down, new_up, keep, error


def resize_file(source: Path, target: Path, max_rank: int, frac: float, dtype: torch.dtype, device: str) -> None:
    modules, metadata, layout = read_lora(source)
    down_name, up_name = layout.split("/")
    original = sorted({down.shape[0] for down, _, _ in modules.values()})

    state: Dict[str, torch.Tensor] = {}
    ranks: List[int] = []
    errors: List[float] = []
    for name, (down, up, alpha) in modules.items():
        new_down, new_up, keep, error = resize_module(down, up, alpha, max_rank, frac, device)
        state[f"{name}.{down_name}.weight"] = new_down.to(dtype).cpu().contiguous()
        state[f"{name}.{up_name}.weight"] = new_up.to(dtype).cpu().contiguous()
        state[f"{name}.alpha"] = torch.tensor(float(keep), dtype=torch.float32)
        ranks.append(keep)
        errors.append(error)

    capped = sum(1 for rank in ranks if rank == max_rank and max_rank < max(original))
    meta = dict(metadata)
    meta["ss_network_dim"] = str(max(ranks))
    meta["ss_network_alpha"] = str(max(ranks))
    meta["maltrainer_resize"] = json.dumps({
        "method": "SVD per module, sv_fro rule",
        "max_rank": max_rank,
        "frac": frac,
        "original_ranks": original,
        "save_dtype": str(dtype).replace("torch.", ""),
        "source": {"file": source.name, "sha256": sha256_prefix(source)},
    })

    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target), metadata=meta)
    print(f"  {len(ranks)} modules, rank {original} -> {min(ranks)}..{max(ranks)} (mean {sum(ranks) / len(ranks):.1f})")
    print(f"  {capped}/{len(ranks)} modules hit the rank cap" + (" (a higher --rank would keep more)" if capped else ""))
    print(f"  dropped norm per module: mean {sum(errors) / len(errors):.1%}, worst {max(errors):.1%}")
    print(f"  wrote {target} ({target.stat().st_size / 1024 ** 2:.0f} MB, {str(dtype).replace('torch.', '')})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="LoRA .safetensors file(s) to shrink")
    parser.add_argument("--rank", type=int, default=8, help="rank cap per module (default 8)")
    parser.add_argument("--frac", type=float, default=0.999,
                        help="share of each module's Frobenius norm to keep before the cap applies (default 0.999)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="saved dtype (default bfloat16)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--output", type=Path, help="output file (only with a single input)")
    parser.add_argument("--suffix", default="_resized", help="added to the input name by default (default _resized)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    if args.rank < 1:
        parser.error("--rank must be at least 1")
    if not 0.0 < args.frac <= 1.0:
        parser.error("--frac must be in (0, 1]")
    if args.output and len(args.inputs) > 1:
        parser.error("--output works with a single input; use --suffix for several")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    jobs: List[Tuple[Path, Path]] = []
    for source in args.inputs:
        if not source.is_file():
            parser.error(f"not found: {source}")
        target = args.output or source.with_name(f"{source.stem}{args.suffix}{source.suffix}")
        if target.resolve() == source.resolve():
            parser.error(f"{source}: the output would overwrite the input")
        if target.exists() and not args.force:
            parser.error(f"{target} exists; pass --force to overwrite")
        jobs.append((source, target))

    failed = 0
    for source, target in jobs:
        print(f"{source} -> rank <= {args.rank} on {device}")
        try:
            resize_file(source, target, args.rank, args.frac, DTYPES[args.dtype], device)
        except ValueError as error:
            print(f"  error: {error}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
