#!/usr/bin/env python3
"""Convert a bf16 ComfyUI DiT to INT8 ConvRot, laid out like a reference INT8 ConvRot file.

    <ComfyUI python> scripts/convert_int8_convrot.py serenity_bf16.safetensors serenity_int8_convrot.safetensors \
        --reference moss_image_sft_int8_convrot.safetensors

INT8 ConvRot (arXiv:2512.03673) folds a block Hadamard rotation into each weight offline and stores
the rotated weight as int8 with one fp32 scale per output row; ComfyUI rotates the activation the
same way at run time, so the rotation cancels in the matmul. Per quantized Linear the file holds

    <module>.weight        int8  [out, in]
    <module>.weight_scale  fp32  [out, 1]
    <module>.comfy_quant   uint8 JSON {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

The reference decides which Linears are quantized and their marker, and supplies the other blobs
an INT8 release carries (MOSS-Image: each block's `comfy_attention.config`, which selects
comfy-kitchen's int8 attention). Every other tensor is copied from the source as stored. The
quantization itself is comfy-kitchen's own `quantize_int8_convrot_weight`, so run this with the
Python that runs ComfyUI (it has `comfy_kitchen`).
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the bf16 (or fp16 / fp32) ComfyUI single file")
    parser.add_argument("output", type=Path, help="where to write the INT8 ConvRot file")
    parser.add_argument("--reference", type=Path, required=True,
                        help="an INT8 ConvRot file of the same architecture (e.g. Comfy-Org's release)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="where to rotate and quantize (default: cuda if available)")
    parser.add_argument("--force", action="store_true", help="overwrite the output")
    args = parser.parse_args()

    try:
        from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    except ImportError:
        print("comfy_kitchen is missing: run this with ComfyUI's Python", file=sys.stderr)
        return 1
    if args.output.exists() and not args.force:
        print(f"{args.output} exists (pass --force to overwrite)", file=sys.stderr)
        return 1

    started = time.time()
    with safe_open(str(args.reference), framework="pt", device="cpu") as ref:
        ref_keys = set(ref.keys())
        markers = {k[: -len(".comfy_quant")]: ref.get_tensor(k) for k in ref_keys if k.endswith(".comfy_quant")}
        configs = {k: json.loads(bytes(t.tolist()).decode("utf-8")) for k, t in markers.items()}
        extras = {k: ref.get_tensor(k) for k in ref_keys
                  if not k.endswith((".weight", ".weight_scale", ".comfy_quant", ".bias")) and ref.get_slice(k).get_dtype() == "U8"}

    out = {}
    with safe_open(str(args.source), framework="pt", device="cpu") as src:
        metadata = src.metadata() or {}
        src_keys = list(src.keys())
        missing = [m for m in markers if f"{m}.weight" not in src_keys]
        if missing:
            print(f"the reference quantizes {len(missing)} Linears the source does not have, e.g. {missing[:3]}", file=sys.stderr)
            return 1
        for i, key in enumerate(src_keys):
            module = key[: -len(".weight")] if key.endswith(".weight") else None
            if module in configs:
                cfg = configs[module]
                if cfg.get("format") != "int8_tensorwise" or not cfg.get("convrot"):
                    print(f"{module}: reference format {cfg} is not INT8 ConvRot", file=sys.stderr)
                    return 1
                w = src.get_tensor(key).to(args.device, torch.float32)
                q, scale = quantize_int8_convrot_weight(w, int(cfg.get("convrot_groupsize", 256)))
                out[key] = q.cpu().contiguous()
                out[f"{module}.weight_scale"] = scale.float().reshape(-1, 1).cpu().contiguous()
                out[f"{module}.comfy_quant"] = markers[module].clone()
            else:
                out[key] = src.get_tensor(key).contiguous()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(src_keys)} tensors", flush=True)
    for key, blob in extras.items():
        out.setdefault(key, blob.clone())

    tmp = args.output.with_name(args.output.name + ".tmp")
    metadata = dict(metadata, int8_convrot_reference=args.reference.name, int8_convrot_source=args.source.name)
    save_file(out, str(tmp), metadata=metadata)
    tmp.replace(args.output)
    size = args.output.stat().st_size / 1e9
    print(f"wrote {args.output} ({size:.2f} GB): {len(markers)} Linears INT8 ConvRot, {len(extras)} reference blobs, "
          f"{len(out)} tensors in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
