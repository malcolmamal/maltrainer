# Merging checkpoints and LoRAs into a model, and INT8 ConvRot back to fp16

Two scripts for whole model files (ComfyUI single files). Both write the output aside and rename it into place, and
refuse to overwrite without `--force`.

## `scripts/merge_lora_into_model.py`: checkpoints and LoRAs into one model

Blends other checkpoints into a base, adds what a finetune changed, and bakes in LoRAs at their strengths, in one
pass. The result loads in ComfyUI and in the trainers exactly like the base (e.g. as `model.transformer_path` for
the next finetune).

```bash
python scripts/merge_lora_into_model.py base.safetensors merged.safetensors a.safetensors:0.8 b.safetensors:1.2
python scripts/merge_lora_into_model.py base.safetensors merged.safetensors --model other.safetensors:0.3
python scripts/merge_lora_into_model.py turbo.safetensors people_on_turbo.safetensors --add-diff finetune.safetensors,raw.safetensors:1
python scripts/merge_lora_into_model.py --config templates/examples/merge_into_model_krea2.yaml
```

Per tensor of the base A, in float32 (on the GPU when there is one), then stored in the output dtype:

    W = (1 - sum t) * A + sum t * B                  --model B:t          weighted sum (any number of models)
        + sum s * (B - C)                            --add-diff B,C:s     add difference (C = the model B came from)
        + sum strength * (alpha / rank) * up @ down  the LoRAs            (LoRA pairs)
        + sum strength * scale * kron(w1, w2)                             (LoKr)
        + sum strength * D                                                (ComfyUI .diff / .diff_b deltas)

- **Weighted sum** blends checkpoints of the same model, as many as you list, each at its weight; the base keeps
  `1 - sum` of them (three in equal thirds: two models at 0.333). Each is compared with the base, not with the others.
  A 50/50 blend of Kroma 0.3.1 Turbo and Krea2Realism (`config/merge_kroma_realism_people.yaml` was one) still renders
  like Turbo: 10 steps, CFG 1.
- **Add difference** moves only what a finetune changed onto another model, at full rank: `Turbo + (finetune - RAW)`
  puts the finetuned people on Turbo for 8-step renders without the rank cut of an extracted LoRA. C must be the
  exact model B was trained from (for a finetune started from `krea2_raw_fp8_scaled`: that file dequantized to bf16,
  as `extract_lora` uses it).
- **LoRAs**: pairs `lora_A` / `lora_B` (maltrainer, PEFT) or `lora_down` / `lora_up` (kohya) with an optional `alpha`
  (missing = rank), and ComfyUI's full deltas `<module>.diff` / `<module>.diff_b` (`krea2filterbypass3` is one).
  Module names are matched with or without the `diffusion_model.` / `model.diffusion_model.` prefix.
- **LoKr** (LyCORIS Kronecker, e.g. `realism_engine_krea2_v1`): `strength * scale * kron(w1, w2)`, each factor stored
  whole (`lokr_w1`) or as a low-rank pair (`lokr_w1_a` @ `lokr_w1_b`), exactly as ComfyUI's LoKr adapter applies it:
  scale = alpha / rank when a factor is a low-rank pair (w2's rank when both are), and 1.0 when both are whole, where
  ComfyUI ignores alpha (that file stores ~1e10 there). Checked against ComfyUI's own `LoKrAdapter` on a real module:
  identical.

**Progress.** The run prints what goes in (the base and its share, each model, each difference, each LoRA with its
module count), a bar over the base's tensors, a bar over the bytes it hashes for the record, then the distances, the
summary line and any warnings. A Krea 2 merge of two 26 GB models and two LoRAs takes ~2-4 minutes: about 2-3 GB of
VRAM (one tensor at a time) and ~30 GB of system RAM (the output is collected before it is written).

**YAML.** `--config` takes every setting (`base`, `output`, `dtype`, `device`, `models`, `add_difference`,
`loras`); [templates/examples/merge_into_model_krea2.yaml](../templates/examples/merge_into_model_krea2.yaml) is
a commented example (the Krea 2 people finetune on Turbo, with a text-fusion LoRA; its `_base_dequantized_bf16`
input is the fp8 RAW base dequantized as the finetune read it, written next to that run's output). The command line's base,
output, `--dtype` and `--device` override the file's; its `--model`, `--add-diff` and LoRAs are added to its lists.

**Warnings, not errors.** Differences between the models are printed as warnings after the file is written, so you
can check and run again:

- a tensor a model lacks or holds in another shape (the base's value is kept for that term), or one it adds (ignored)
- a model more than 25% away from the base (likely another base model), or a finetune more than 25% away from the
  model given as its base (likely the wrong C); a real finetune moves a few percent (the Krea 2 people run: 2.9%)
- an fp8 file in a difference: its rounding noise is part of `B - C` (bf16 / fp16 is cleaner)
- model weights summing outside [0, 1] (extrapolation)

Refused before anything is written: an INT8 ConvRot model (dequantize it first, below), a scale on anything but an
fp8 weight, a LoRA with tensors it does not understand (DoRA, LoHa, convolutional LoKr), a LoRA module the base
lacks, or a LoRA delta of the wrong shape.

**The base and the dtype.** bf16, fp16, fp32 or fp8-scaled models (dequantized on the way, so the output is plain,
without `_quantization_metadata`). `--dtype` defaults to the base's own; an fp8 base writes bfloat16.

**The record.** The base's metadata is kept and `maltrainer_merge_into_model` added: the base, every model and LoRA
with a SHA-256 prefix, their weights and strengths, and the saved dtype. The hashes are why it reads every input
once more at the end: a file name can be reused (a finetune's `<name>.safetensors` is overwritten by its next run),
the hash says which file it really was.

To merge LoRAs into one LoRA instead (lossless, rank concatenation), use [merge_loras.py](merge-loras.md).

## `scripts/dequantize_int8_convrot.py`: INT8 ConvRot back to a plain file

The reverse of `scripts/convert_int8_convrot.py`. Run it with ComfyUI's Python (it needs `comfy_kitchen`).

```bash
C:\Development\ComfyUI\venv\Scripts\python.exe scripts/dequantize_int8_convrot.py Krea2Realism_V1_INT8_ConvRot.safetensors Krea2Realism_V1_fp16.safetensors
```

Each Linear marked INT8 ConvRot (`<module>.comfy_quant`) is scaled by its per-row `weight_scale` and rotated back
from the block-Hadamard basis with comfy-kitchen's own `dequantize_int8_convrot_weight`: the weights ComfyUI
computes with when it runs the INT8 file, so a render at the same seed matches it. The INT8 rounding stays (on a
Krea 2 finetune: within 0.9% of the original bf16 weights); this recovers the format, not the precision. Every
other floating tensor is cast to `--dtype` (default float16), the markers and scales are dropped, and any other
quantization format is refused.

Uses: a base the trainers and `extract_lora.py` accept (both refuse INT8), or a base for `merge_lora_into_model.py`.
