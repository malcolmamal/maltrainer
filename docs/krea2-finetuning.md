# Krea 2 full finetuning

`train_krea2_finetune.py` trains the Krea 2 DiT's own weights instead of a LoRA, on one 32 GB
card. Data, captions, timesteps and loss are the LoRA trainer's
([krea2-lora-training.md](krea2-lora-training.md)); the walkthrough for finetunes in general is
[finetune-tutorial.md](finetune-tutorial.md).

```bash
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml --dry-run
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml --resume-from output/krea2_finetune/people/krea2_ft_people_state
```

Start from `templates/krea2_finetune_template.yaml`.

> **Status (2026-10-19):** built and tested on the CPU with a 3-block Krea 2 (windows rotate in
> order, frozen weights stay bit-identical, a resumed run ends bit-identical to an uninterrupted
> one, saved files load back). **Not yet run on the GPU at full size**: the VRAM and speed below
> are estimates until a probe run measures them.

## Why windows

Krea 2 is 12.9B parameters. Its bf16 weights alone are 25.8 GB; a naive full finetune (weights,
gradients and optimizer state at once) needs about 78 GB (Fizgig's figure). So the blocks train
a window at a time, as Fizgig does (`fizgig/families/ft.py`):

| Window | What | Parameters |
|---|---|---|
| `attention` | `blocks.N.attn.{wq,wk,wv,gate,wo}`, all 28 blocks | 3.7B |
| `mlp.gate` | `blocks.N.mlp.gate`, all 28 blocks | 2.8B |
| `mlp.up` | `blocks.N.mlp.up` | 2.8B |
| `mlp.down` | `blocks.N.mlp.down` | 2.8B |
| always on | the text-fusion stack (`txtfusion`) | 0.35B |

- **The frozen model stays on the GPU in fp8**, per output row, as for Krea 2 LoRAs. Fizgig uses
  4-bit NF4 to fit 16 GB cards; fp8 is closer to the bf16 weights and still fits 32 GB.
- **The active window is bf16**, copied from a **master** in CPU RAM. Every `rotate_every` steps
  it is written back to the master, re-quantized to fp8 for the other windows' forward passes, and
  the next window comes up. Training never round-trips through fp8: the master is the truth.
- **Each window spans every block**, so a concept is learned at every depth at once.
- **The text-fusion stack trains all run.** It is where prompts bind to concepts, and a rotation
  over the blocks would never reach it (Fizgig's `always_on`). Norms, modulation, biases and the
  input/output layers stay frozen.
- **Each window keeps its optimizer state** in CPU RAM while frozen. Fizgig rebuilds its optimizer
  every window; here a rotation continues where the window left off, and a resume is exact.

The windows and the interval are in the config:

```yaml
finetune:
  rotate_every: 100
  windows:
    - {name: attention, prefixes: ["attn."]}
    - {name: mlp.gate, prefixes: ["mlp.gate"]}
    - {name: mlp.up, prefixes: ["mlp.up"]}
    - {name: mlp.down, prefixes: ["mlp.down"]}
  always_on: ["txtfusion"]
```

The window at step `s` is `(s // rotate_every) mod windows`, so a `max_steps` that is a multiple
of `4 × rotate_every` trains every window equally. A prefix matches the Linear's name inside its
block; a Linear in no window stays frozen (with a warning). `rotate_every` must be at least 1; leave it out (or null)
for the default of 100. A 0 or a negative value stops the run at start with an error (it used to be read as 100).

## Memory

**Estimated** for 512 px (not measured yet):

| Part | Size |
|---|---|
| Frozen model in fp8 (block Linears) + bf16 text fusion and I/O | ~13.5 GB |
| Active window in bf16, minus its own fp8 copy | +3.7 GB (attention), +2.8 GB (an MLP window) |
| Automagic3 sign history (1 byte per trained weight at `polarity_history: 8`) | +3.7 GB / +2.8 GB |
| Activations with gradient checkpointing | ~1-2 GB |
| **Peak** | **~22-24 GB**, attention window |

For comparison, Fizgig measured ~15.6 GB at ~0.85 s/step at 0.25 MP on a 32 GB card, with an NF4
trunk about half the size of the fp8 one. **It does not fit beside another training run**: a
MOSS-Image finetune alone takes ~25 GB.

CPU RAM: the master (25.8 GB; from a bf16 file it is the file's own tensors, no copy) plus every
frozen window's optimizer state (~12.5 GB in total), ~40 GB.

Disk: each checkpoint is a full bf16 model (~26 GB). The resume state hard-links it and adds the
optimizer states (~12.5 GB).

## The base model

Finetune the **bf16 Raw** model (`krea2_raw_bf16.safetensors`, ~26 GB). Turbo is distilled.
An fp8 file (`krea2_raw_fp8_scaled`) loads too, with a warning: its blocks are dequantized into
the master, so the finetune starts from fp8's rounding, and every saved file carries it. Use it to
test the setup, not for a real run. LoRA extraction also needs the bf16 base.

## Optimizer and rate

Fused Automagic3 only: each weight steps inside backward as soon as its gradient exists, so no
full gradient set is held, and the update is rounded stochastically into the bf16 weight, so small
steps are not lost. `gradient_accumulation_steps` must be 1, and there is no global gradient clip
(`automagic_clip_threshold` clips each tensor's update instead).

The template pins the rate at **1e-5** (`learning_rate` = `automagic_min_lr` = `automagic_max_lr`),
Fizgig's Krea 2 finetune rate. Fizgig: 1e-4 "trains, but treat it as the top of the experiment
range"; when a run looks almost right but overcooked, lower the rate rather than the step count.

## Output and resume

- **Checkpoints** are full bf16 ComfyUI files with the source's key prefix: load them in ComfyUI
  like `krea2_raw_bf16`, or as `model.transformer_path` here (for LoRAs or more finetuning).
- **Keys:** `s` save and continue, `q` save and stop, `p` pause, `h` help.
- **Resume** with `--resume-from <name>_state`: the weights, every window's optimizer state, the
  step (so the window), the position in the epoch and the RNG. Verified bit-exact on the CPU test.
  Raise `max_steps` to train past the old end. Keep `finetune.windows` as it was; the trainer
  refuses a different set.
- **A changed dataset** on resume starts a fresh epoch; the rest carries on.
- **As a LoRA:** `python scripts/extract_lora.py krea2_raw_bf16.safetensors <finetune>.safetensors --rank 64`.
  Fizgig found rank 64 indistinguishable from the full checkpoint by eye.

## Not there yet

Compared with the MOSS-Image finetune ([moss-image-finetuning.md](moss-image-finetuning.md)):

- no previews during training (the Krea 2 trainers have no sampler yet)
- no prior preservation, regularisation images or per-folder repeats (`train.dop` is refused)
- no EMA (an fp32 average of 12.9B weights would be 52 GB of CPU RAM)
- no `--suggest-steps` and no on-disk encoder cache: every start encodes the dataset again
- one GPU only; batch size 1
