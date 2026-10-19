#!/usr/bin/env python3
"""Boost a Krea 2 LoRA's four text-fusion blocks (Fizgig's Repair Studio "Text fusion x2 / x3").

Krea 2 mixes the text encoder's layers through a small text-fusion transformer
(``txtfusion.layerwise_blocks.0/1`` and ``txtfusion.refiner_blocks.0/1``) before the 28 DiT
blocks see the prompt. Fizgig measured, across several Krea 2 LoRAs, that multiplying only those
four blocks' updates by 3 raised its detail metric (~72 -> ~77) and likeness by 2-7 points with
the composition unchanged. The one exception was an overtrained LoRA, which it made worse.

This multiplies lora_B / lora_up of every adapter inside those blocks by ``--factor``, so their
weight change is ``factor`` times larger and everything else is untouched. The result works
anywhere at strength 1.0. ``txtfusion.projector`` and the DiT blocks are left alone, as in Fizgig.

    python scripts/boost_text_fusion.py a.safetensors                 -> a_txt3.safetensors (x3)
    python scripts/boost_text_fusion.py a.safetensors --factor 2      -> a_txt2.safetensors
    python scripts/boost_text_fusion.py a.safetensors b.safetensors --factor 2

Reads maltrainer / ComfyUI keys (``diffusion_model.txtfusion.refiner_blocks.0.attn.wq.lora_B``)
and kohya keys (``lora_unet_txtfusion_refiner_blocks_0_attn_wq.lora_up``). A LoRA trained with
``lora.targets: attn-mlp`` has no text-fusion adapters and is refused. The input is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

from safetensors import safe_open
from safetensors.torch import save_file

from merge_loras import sha256_prefix

# the four blocks, in dotted (ComfyUI) or underscored (kohya) spelling
TEXT_FUSION_BLOCK = re.compile(r"txtfusion[._](layerwise|refiner)[._]blocks[._](\d)")
UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight")


def boost_file(source: Path, target: Path, factor: float) -> None:
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        state = {key: handle.get_tensor(key) for key in handle.keys()}

    boosted = []
    for key, tensor in state.items():
        if key.endswith(UP_SUFFIXES) and TEXT_FUSION_BLOCK.search(key):
            state[key] = (tensor.float() * factor).to(tensor.dtype).contiguous()
            boosted.append(key)
    if not boosted:
        raise ValueError(
            f"{source.name}: no text-fusion block adapters (txtfusion.layerwise_blocks / refiner_blocks); "
            "was it trained with lora.targets attn-mlp, or is it not a Krea 2 LoRA?"
        )
    blocks = sorted({"txtfusion.{}_blocks.{}".format(*TEXT_FUSION_BLOCK.search(key).groups()) for key in boosted})

    meta = dict(metadata)
    meta["maltrainer_text_fusion_boost"] = json.dumps(
        {
            "factor": factor,
            "blocks": blocks,
            "adapters": len(boosted),
            "source": {"file": source.name, "sha256": sha256_prefix(source)},
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target), metadata=meta)
    print(f"  x{factor:g} on {len(boosted)} adapters in {len(blocks)} blocks: {', '.join(blocks)}")
    print(f"  wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Krea 2 LoRA .safetensors file(s)")
    parser.add_argument("--factor", type=float, default=3.0, help="multiplier on the text-fusion blocks (default 3)")
    parser.add_argument("--output", type=Path, help="output file (only with a single input)")
    parser.add_argument("--suffix", default=None, help="added to the input name (default _txt<factor>)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    if args.factor <= 0:
        parser.error("--factor must be positive")
    if args.output and len(args.inputs) > 1:
        parser.error("--output works with a single input; use --suffix for several")
    suffix = args.suffix if args.suffix is not None else f"_txt{args.factor:g}"

    jobs: List[Tuple[Path, Path]] = []
    for source in args.inputs:
        if not source.is_file():
            parser.error(f"not found: {source}")
        target = args.output or source.with_name(f"{source.stem}{suffix}{source.suffix}")
        if target.resolve() == source.resolve():
            parser.error(f"{source}: the output would overwrite the input")
        if target.exists() and not args.force:
            parser.error(f"{target} exists; pass --force to overwrite")
        jobs.append((source, target))

    failed = 0
    for source, target in jobs:
        print(f"{source}")
        try:
            boost_file(source, target, args.factor)
        except ValueError as error:
            print(f"  error: {error}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
