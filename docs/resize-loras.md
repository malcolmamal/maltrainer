# Resizing a LoRA: `scripts/resize_lora.py`

`resize_lora.py` shrinks a LoRA to a lower rank, for example rank 32 to rank 8. The file is
about four times smaller and loads and runs the same way in ComfyUI. The input is never
modified; the result is written next to it as `<name>_resized.safetensors`.

Unlike [merging](merge-loras.md), resizing is lossy: it keeps the strongest directions of each
layer and drops the rest. The script prints how much it dropped.

## Quick start

From the repo root, with the project venv:

```bash
python scripts/resize_lora.py output/my_lora/my_lora_step_500.safetensors
```

That writes `my_lora_step_500_resized.safetensors` at rank 8. On Windows you can also drag
`.safetensors` files onto `resize_lora.bat`.

More examples:

```bash
python scripts/resize_lora.py a.safetensors b.safetensors --rank 16     # several files, rank 16
python scripts/resize_lora.py a.safetensors --output out/a_r8.safetensors
python scripts/resize_lora.py a.safetensors --dtype float32             # keep float32 precision
```

## Options

| Option     | Default      | Meaning                                                                 |
| ---------- | ------------ | ----------------------------------------------------------------------- |
| `--rank`   | `8`          | most singular values kept per layer                                     |
| `--frac`   | `0.999`      | keep only as many as hold this share of the layer's norm, if fewer than `--rank` |
| `--dtype`  | `bfloat16`   | saved dtype: `float32`, `bfloat16`, `float16`                           |
| `--device` | cuda if any  | `cuda` or `cpu`                                                         |
| `--output` | -            | output file, for a single input                                         |
| `--suffix` | `_resized`   | appended to the input name by default                                   |
| `--force`  | off          | overwrite an existing output                                            |

## How it works

For one layer the LoRA applies `W = (alpha / rank) * B @ A`. The script keeps the fewest
singular values of `W` that hold `--frac` of its Frobenius norm (the rule of kohya's
`resize_lora.py --dynamic_method sv_fro`), at most `--rank` of them, and writes them back as a
smaller `lora_A` / `lora_B` pair with `alpha` set to the new rank.

The SVD runs on a small rank x rank core, not on the full matrix. `B` and `A^T` are QR-factored,
so `W = Qb @ (Rb @ Ra^T) @ Qa^T`, and the singular values of `W` are those of the core. This is
exact and takes seconds for a whole LoRA, where a plain SVD of every full matrix took minutes.
It also gives the error for free.

## Reading the output

```
252 modules, rank [32] -> 8..8 (mean 8.0)
252/252 modules hit the rank cap (a higher --rank would keep more)
dropped norm per module: mean 33.3%, worst 58.0%
```

- **Hit the rank cap** means `--frac` would have kept more than `--rank`. When most modules hit
  it, the LoRA has real content beyond that rank.
- **Dropped norm** is the share of each layer's update that was thrown away, as
  `sqrt(sum of dropped s^2 / sum of all s^2)`. The numbers above are typical of a rank-32
  character LoRA cut to 8; that is a big cut, so compare the result with the original on a few
  seeds. If it looks weaker, use `--rank 16`.

The output keeps the input's metadata, with `ss_network_dim` / `ss_network_alpha` set to the new
rank and a `maltrainer_resize` entry recording the settings and the source file's hash.

## Requirements

Only what the project venv already has: `torch` (CUDA is optional, the script is fast on CPU too
for a few-hundred-module LoRA, just slower), `safetensors`. No extra install. It reuses
`read_lora` from `merge_loras.py`, so it takes the same key layouts (`lora_A`/`lora_B`,
`lora_down`/`lora_up`, optional `.alpha`) and refuses what it does not understand (conv layers,
DoRA, LoKr) instead of dropping it.

## Not kohya's script

Stabilis, the backend that runs the trainers, has its own post-train "Resize LoRA" option and a
Resize tab, which call kohya's `resize_lora.py` and need a kohya venv with torch. This script
does not depend on kohya. It gives the same result up to rounding, so use it when that venv is
not available.
