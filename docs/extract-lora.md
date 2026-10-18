# Extracting a LoRA from a finetune: `scripts/extract_lora.py`

`extract_lora.py` turns a full finetune into a LoRA. It takes each layer's change against the
base model and keeps its strongest directions with an SVD. This is the DreamBooth-then-extract
workflow (kohya's `extract_lora_from_models.py`, LyCORIS's `extract_locon`).

The result is an ordinary LoRA in ComfyUI's keys. Load it on the **base** model with the stock
LoRA loader, stack it with other LoRAs, or [resize](resize-loras.md) and [merge](merge-loras.md)
it like a trained one. A 10 GB MOSS-Image finetune becomes a few hundred MB.

Fizgig reports two things for this workflow:
- rank 64 looked the same as the full checkpoint
- an extracted LoRA beat a LoRA trained directly at the same rank on the same data, because the
  full-rank training finds a better solution and the SVD keeps most of it

## Quick start

From the repo root, with the project venv:

```bash
python scripts/extract_lora.py base.safetensors finetuned.safetensors
```

That writes `finetuned_lora_r64.safetensors` next to the finetune. On Windows you can also run
`extract_lora.bat base.safetensors finetuned.safetensors`.

Several ranks cost one pass, since one SVD per layer serves them all:

```bash
python scripts/extract_lora.py base.safetensors ft.safetensors --rank 16 32 64 128
```

## Choosing each layer's rank

| `--mode` | Keeps per layer | `--value` |
|---|---|---|
| `fixed` (default) | exactly `--rank` | - |
| `fro` | the fewest singular values holding `--value` of the layer's change (kohya's `sv_fro`, LyCORIS's `quantile`) | e.g. `0.9` |
| `ratio` | singular values ≥ `--value` × the layer's largest (LyCORIS's `ratio`) | e.g. `0.05` |
| `threshold` | singular values ≥ `--value` (LyCORIS's `threshold`) | absolute |

The dynamic modes stay under `--rank`, which then acts as a cap. They give layers the finetune
barely touched a small rank and heavily changed layers more, which is usually a better size for
the same quality than one flat rank:

```bash
python scripts/extract_lora.py base.safetensors ft.safetensors --mode fro --value 0.9 --rank 128
```

For each output, the script prints the rank range and how much of each layer's change was
dropped (the mean and the worst layer). A high "worst" means the cap is cutting into a layer
that changed a lot.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--rank` | `64` | rank, or the cap in a dynamic mode; several values write several files |
| `--mode` | `fixed` | see above |
| `--value` | - | the threshold of a dynamic mode |
| `--model` | `auto` | checkpoint layout: `moss_image`, `generic`, or detected |
| `--dtype` | `bfloat16` | saved dtype: `float32`, `bfloat16`, `float16` |
| `--device` | cuda if any | where the SVDs run |
| `--output-dir` | next to the finetune | where the files go |
| `--name` | `<finetune>_lora` | output name stem; `_r<rank>` is appended |
| `--force` | off | overwrite existing outputs |

## What it extracts

- Only 2-D weights (Linear layers). A changed norm scale or bias is listed and left out; a LoRA
  cannot hold it. MOSS-Image finetunes with `finetune.linears_only: true` (the default) change
  only Linear weights, so nothing is lost there.
- Unchanged tensors are skipped, so a finetune of a few blocks gives a LoRA of those blocks only.
- MOSS-Image: ComfyUI's files fuse each block's q/k/v into `proj.qkv`. The script splits the
  change into `proj.q` / `proj.k` / `proj.v` LoRA modules, each with its own SVD. That is the
  same layout trained MOSS-Image LoRAs use: ComfyUI slices it into the fused weight.
- Both files must be plain bf16/fp16/fp32 checkpoints with the same tensors, such as
  `moss_image_base_bf16.safetensors` and a finetune this repo saved from it. Int8 files are
  refused.

The code is split between the model-agnostic SVD (`maltrainer/finetune/extract.py`) and each
model's layout (MOSS-Image's in `maltrainer/models/moss_image/finetune.py`).
