# MiniMax H3 Image full finetuning

`train_minimax_h3_image_finetune.py` trains the DiT's own weights instead of a LoRA, from ComfyUI's **INT8 ConvRot**
file, on one 32 GB card. Captions, noise levels and loss are the LoRA trainer's
([minimax-h3-image-lora-training.md](minimax-h3-image-lora-training.md)); the finetune rotates windows of blocks the
way the [Krea 2 finetune](krea2-finetuning.md) does, over an INT8 model instead of an fp8 one.

```bash
python train_minimax_h3_image_finetune.py --config my_run.yaml
python train_minimax_h3_image_finetune.py --config my_run.yaml --dry-run
python train_minimax_h3_image_finetune.py --config my_run.yaml --resume-from output/.../<name>_state
```

Start from `templates/minimax_h3_image_finetune_template.yaml`.

> **Status (12 November 2026): not yet run on the GPU.** The tests run the whole trainer on a tiny CPU model with a real
> INT8 file (windows rotate, the saved file loads, a resumed run ends bit for bit where an uninterrupted one does). On
> the real 21 GB file, on the CPU, the loader, the 200 swappable Linears, Fizgig's adapter (200 modules) and one
> window's swap in and out were checked. Memory on the 5090 is estimated from the file's sizes: do a short first run
> (below) before a long one. Fizgig has H3 Image finetuning switched off ("not done for H3-Image yet"), so its
> learning rate is the H3 one (3e-5), not a tested H3 Image one.

## Why windows

H3 Image is 19.3B parameters in 50 blocks of ~385M (qkv 116M, out 39M, fc1 154M, fc2 77M), stored as INT8 (21 GB). Its
bf16 weights alone would be 39 GB. So the blocks train a window at a time:

- **The frozen model stays on the GPU as INT8 codes**, as for LoRAs (about 21 to 22 GB).
- **One window is trainable in bf16**: `finetune.blocks_per_window` blocks (default 5, 1.9B parameters, 3.9 GB bf16
  plus 1.9 GB of Automagic3 state). `finetune.rotate_every` steps later (default 100) it is quantized back to INT8 and
  the next window comes up. A full cycle is 10 windows.
- **The bf16 value of every weight a window has trained** is kept in CPU RAM (up to 39 GB), with each window's
  optimizer state (up to 19 GB at `automagic_polarity_history: 8`). Plan for about 60 GB of RAM at the end of the
  first cycle.
- **Training never round-trips through INT8**: a window starts from its bf16 value and writes back to it. Only the
  forward passes of the other windows see the INT8 codes of its latest value.
- **A weight is only re-quantized once a window has visited it.** Re-encoding a decoded weight is not bit-exact (on
  one block of the real file 97 to 99 % of the codes came back the same, none differed by more than one step), so a
  weight that is visited but barely trained moves by up to one INT8 step.
- **Backward** stops at the first window that trains, so a `block_range` over the back of the stack is faster.

Each window's step is **fused Automagic3**: every weight steps inside backward and the update is rounded
stochastically into bf16. There is no gradient clipping (`max_grad_norm` is not used) and no gradient accumulation.

## What trains

```yaml
finetune:
  rotate_every: 100
  blocks_per_window: 5
  block_range: [20, 49]     # optional: only these blocks; the others stay INT8 and untouched
  targets: ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]   # the default
  always_on: []             # modules trained all run, e.g. ["final_layer.video_out"]
  training_adapter: true
```

The token refiner, the input and output layers, the norms and the AdaLN stay frozen unless `always_on` names their
modules (`bf16` or `fp32` Linears only; the AdaLN is stored in fp16).

## Fizgig's training adapter: on or off

`finetune.training_adapter` (default `true`) switches the adapter named by `model.training_adapter_path`.

- **On**: it rides frozen over the Linears (multiplier `finetune.training_adapter_multiplier`, 1.0), as in the LoRA
  recipe, so the weights are trained against the model *with* the adapter. It is not in the saved file, and the
  finetuned model runs without it, as a LoRA trained this way does.
- **Off**: the weights are trained against the plain distilled model, and the adapter file is not read.

Which is better for a finetune is not measured here; the LoRA recipe needs it ("without it, people trained on this
model turn into caricatures"). Try both on a short run.

## Saved files

- `<name>.safetensors` (and `<name>_step_N.safetensors`): **an INT8 ConvRot file of the original's size (~21 GB)**,
  written from the original's tensor list, so ComfyUI loads it as it loads the original. Its metadata keeps the
  original's noise-level grid. It can be the `transformer_path` of a LoRA run or the next finetune.
- `<name>_state/` (with `train.save_state`, the latest save only): `weights.safetensors` (the bf16 value of every
  trained weight), `optimizer.pt` and `trainer_state.pt`. Up to 39 GB + 19 GB.

A save waits for free disk space when it is the final one, and is skipped (training goes on) when it is a numbered one.
`train.save_every: 0` saves only at the end.

## Resume

`--resume-from <name>_state` (or `train.resume_from`) continues bit for bit: the epoch's order, every RNG state, the
likeness epoch and each window's optimizer state come back. Keep `finetune.blocks_per_window`, `block_range` and
`targets` as they were. A changed dataset resumes at the saved step with a fresh epoch. A state needs the original
INT8 file at `model.transformer_path`.

## A first run on the GPU

Run a few steps before a long run: `train.max_steps: 20`, `finetune.rotate_every: 5`, `train.save_every: 0`. It
exercises the model load, four window swaps and the final save, and the log shows each swap's time and the peak memory
in Task Manager. If the card runs out of memory, lower `finetune.blocks_per_window` (4 gives 1.5B windows).

## Not done

Regularisation images, prior preservation, previews, per-folder repeats and the control keys of the MOSS-Image and
Krea 2 finetunes; an EMA; saving a bf16 model.
