#!/usr/bin/env python3
"""Convert an INT8 ConvRot ComfyUI DiT back to a plain fp16 (or bf16 / fp32) single file.

    <ComfyUI python> scripts/dequantize_int8_convrot.py model_int8_convrot.safetensors model_fp16.safetensors

The reverse of convert_int8_convrot.py. Each Linear the file marks INT8 ConvRot (``<module>.comfy_quant``) is
scaled by its per-row ``weight_scale`` and rotated back from the block-Hadamard basis with comfy-kitchen's own
``dequantize_int8_convrot_weight`` -- the weights ComfyUI computes with when it runs the INT8 file, so a render
at the same seed matches. The INT8 rounding stays: this recovers the format, not the original precision. Every
other floating tensor is cast to the output dtype; the markers and scales are dropped. Run it with the Python
that runs ComfyUI (it has ``comfy_kitchen``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _marker(handle: Any, keys: set[str], module: str) -> dict[str, Any] | None:
    """The module's INT8 ConvRot settings, None when it is not quantized; refuses any other quantization."""
    marker = f"{module}.comfy_quant"
    if marker not in keys:
        return None
    cfg: dict[str, Any] = json.loads(bytes(handle.get_tensor(marker).tolist()).decode("utf-8"))
    if cfg.get("format") != "int8_tensorwise" or not cfg.get("convrot"):
        raise ValueError(f"{module}: {cfg} is not INT8 ConvRot")
    return cfg


def dequantize_file(
    source: Path,
    dtype: torch.dtype,
    device: str,
    dequantize: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str], int]:
    """(tensors, metadata, how many Linears were dequantized)."""
    out: dict[str, torch.Tensor] = {}
    count = 0
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = dict(handle.metadata() or {})
        for key in sorted(keys):
            if key.endswith((".weight_scale", ".comfy_quant")):
                continue
            cfg = _marker(handle, keys, key[: -len(".weight")]) if key.endswith(".weight") else None
            if cfg is not None:
                module = key[: -len(".weight")]
                q = handle.get_tensor(key).to(device)
                scale = handle.get_tensor(f"{module}.weight_scale").to(device)
                out[key] = dequantize(q, scale, int(cfg.get("convrot_groupsize", 256))).to(dtype).cpu().contiguous()
                count += 1
            else:
                tensor = handle.get_tensor(key)
                out[key] = (tensor.to(dtype) if tensor.is_floating_point() else tensor).contiguous()
    return out, metadata, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the INT8 ConvRot ComfyUI single file")
    parser.add_argument("output", type=Path, help="where to write the plain file")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16", help="output dtype (default float16)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where to dequantize (default: cuda if available)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite the output")
    args = parser.parse_args()

    try:
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    except ImportError:
        print("comfy_kitchen is missing: run this with ComfyUI's Python", file=sys.stderr)
        return 1
    if args.output.exists() and not args.force:
        print(f"{args.output} exists (pass --force to overwrite)", file=sys.stderr)
        return 1

    started = time.time()
    try:
        out, metadata, count = dequantize_file(
            args.source, DTYPES[args.dtype], args.device, dequantize_int8_convrot_weight
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if count == 0:
        print(f"{args.source.name} has no INT8 ConvRot Linears (no .comfy_quant markers)", file=sys.stderr)
        return 1
    tmp = args.output.with_name(args.output.name + ".tmp")
    save_file(out, str(tmp), metadata=dict(metadata, int8_convrot_dequantized_from=args.source.name))
    tmp.replace(args.output)
    size = args.output.stat().st_size / 1e9
    print(
        f"wrote {args.output} ({size:.2f} GB): {count} Linears dequantized to {args.dtype}, "
        f"{len(out)} tensors in {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
