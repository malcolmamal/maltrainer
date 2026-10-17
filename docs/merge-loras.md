# Merging LoRAs: `scripts/merge_loras.py`

`merge_loras.py` turns several LoRAs into **one file that applies exactly what loading them
together does**. In ComfyUI that means one LoRA loader instead of a chain. The result does not
depend on loader order, and you have a single file to keep or share.

The merge is lossless. It does not approximate (no SVD, no rank reduction); it rewrites the sum
of the LoRAs as one bigger LoRA.

## Quick start

Merge the two feliciaday MOSS-Image LoRAs, each at strength 1.0, and check the result:

```bash
python scripts/merge_loras.py output/moss_image_feliciaday_soup.safetensors output/moss_image_feliciaday_automagic3/moss_image_feliciaday_v1_automagic3_step_500.safetensors output/moss_image_feliciaday_prodigy/moss_image_feliciaday_v1_prodigy_step_750.safetensors --verify
```

Then load `moss_image_feliciaday_soup.safetensors` in ComfyUI at **strength 1.0**.

## Strengths

Give each input the strength it had in your stack by appending `:strength` to its path. Without
one, the strength is 1.0.

```bash
python scripts/merge_loras.py out.safetensors automagic_step_500.safetensors:0.8 prodigy_step_750.safetensors:1.1
```

The strengths are folded into the merged file, so load it at 1.0. Strength 1.0 on the merged
file equals the stack at the strengths you merged with. Strength 0.5 equals the whole stack
scaled by half. Windows paths with a drive letter work: `C:\loras\a.safetensors:0.8`.

## Options

| Option | Effect |
|---|---|
| `--verify` | For every module, compares the merged weight change with the sum of the inputs' changes in float64 and prints the worst relative difference |
| `--dtype float32` | Default. Exact: ~1e-8 relative per module |
| `--dtype bfloat16` | Half the file size, ~2e-3 relative rounding. Usually invisible, but not lossless |
| `--force` | Overwrite an existing output file |

## Size

The merged rank is the sum of the input ranks, so two rank-16 LoRAs give one rank-32 LoRA. For
MOSS-Image (`blocks` targets) that is about 300 MB in float32 and 150 MB in bfloat16. Each input
is about 75 MB.

## How it works

ComfyUI applies a LoRA to a layer as a weight change, and stacking LoRAs adds their changes.
LoRA *i* contributes

```
delta_i = strength_i × (alpha_i / rank_i) × B_i · A_i
```

The script stacks the down matrices (A) and sets the up matrices (B) side by side, folding each
LoRA's factor `c_i = strength_i × alpha_i / rank_i` into its part of B. It then sets
alpha = total rank, so ComfyUI's scale is 1:

```
A = [A_1; A_2; …]          B = [c_1·B_1, c_2·B_2, …]          B · A = Σ c_i · B_i · A_i
```

That is the stacked sum, exactly. The inputs may have different ranks and alphas. A layer that
only some inputs train is merged from those inputs.

## Checked against ComfyUI

On a small MOSS-Image model in ComfyUI, two LoRAs were stacked at strengths 0.8 and 1.3, with
different ranks (16 and 8) and alphas (16 and 4). That stack was compared with the merged file
at strength 1.0. The model outputs matched to 1e-7, which is float32 rounding.

## What it accepts

- **Key layouts.** `<module>.lora_A.weight` / `lora_B.weight` (maltrainer, PEFT/diffusers) or
  `<module>.lora_down.weight` / `lora_up.weight` (kohya), each with an optional
  `<module>.alpha`. A missing alpha means alpha = rank.
- **Same model.** The inputs must be for the same model. Layer shapes are checked, and a
  base-model mismatch in the metadata gives a warning.
- **Plain linear LoRAs only.** Files with anything else (DoRA magnitudes, LoKr, LoHa, conv
  LoRAs) are refused rather than merged incorrectly.

## Metadata

The merged file records:

- **`ss_network_dim` / `ss_network_alpha`:** the merged rank.
- **`ss_base_model_version` / `ss_network_module`:** copied when all inputs agree.
- **`maltrainer_merge`:** for each input, the file name, a SHA-256 prefix, the strength, the
  optimizer and its full training config (`maltrainer_config`). A merged file therefore still
  says how each part was trained.

`python scripts/inspect_lora_metadata.py <file>` shows it.
