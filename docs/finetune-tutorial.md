# Finetuning with maltrainer: a walkthrough

This guide takes you from nothing to a finished full finetune: install, model files, dataset,
config, the run itself, resuming, and using the result. It ends with a command cheat sheet and a
troubleshooting table.

Every setting is explained in full in the reference,
[moss-image-finetuning.md](moss-image-finetuning.md). This page links to it instead of repeating
it.

**Which models?** This walkthrough is for MOSS-Image (`train_moss_image_finetune.py`). Krea 2
has a finetune too (`train_krea2_finetune.py`), which trains in rotating windows to fit 12.9B
parameters on 32 GB: see [krea2-finetuning.md](krea2-finetuning.md). The other models
(FLUX.3 [klein], Qwen-Image 2.1, Z-Image, MiniMax H3) train LoRAs only. [Other
models](#other-models) at the end covers what is shared and what a new finetune trainer would
need.

---

## 0. Finetune or LoRA?

| | LoRA (`train_moss_image_lora.py`) | Full finetune (`train_moss_image_finetune.py`) |
|---|---|---|
| Trains | small adapters (rank 16-64) | the model's own weights, 4.76B by default |
| Good for | one person, one style | many people at once, a lasting change to the model |
| Output | ~100-300 MB LoRA | ~10 GB full model per checkpoint (can be [turned into a LoRA](#9-use-the-result)) |
| Optimizer | AdamW 8-bit, Prodigy or Automagic3 | Automagic3 only (fused) |
| Disk per save | small | 10 GB model + resume state (15 GB, or 35 GB with EMA) |
| Typical rate | 1e-4 | 1e-5, pinned |

For a single person, train a LoRA ([moss-image-lora-training.md](moss-image-lora-training.md)).
A finetune pays off when you train many people together (the `people` run trains 30), or when a
LoRA runs out of capacity and the people start to blend.

---

## 1. What you need

**Hardware** (measured on an RTX 5090, 32 GB):

| | Need |
|---|---|
| GPU | 32 GB class. 512 px with all blocks: ~22 GB without extras; the `people` run (EMA + prior preservation) peaks at ~25 GB |
| CPU RAM | the cached latents and captions, plus 30 GB with EMA on (a 20 GB fp32 average + a 10 GB pinned copy buffer), plus ~9.5 GB while recording prior for people added mid-run |
| Disk | 10 GB per checkpoint, plus the latest resume state: ~5 GB extra without EMA (it hard-links the model), ~35 GB with EMA |
| GPUs | one. Multi-GPU is refused ("MOSS-Image finetuning runs on one GPU") |

**Model files** (the ComfyUI ones; the same files ComfyUI loads):

| File | Folder under ComfyUI `models/` | Note |
|---|---|---|
| `moss_image_base_bf16.safetensors` | `diffusion_models/` | **bf16 Base only.** The int8 file is refused, and SFT is the wrong starting point |
| `flux2_vae.safetensors` | `vae/` | the full FLUX.2 VAE |
| `qwen3.5_9b_int8_convrot.safetensors` | `text_encoders/` | used only while encoding captions, then unloaded |

---

## 2. Install

From the repository root, in a terminal (Git Bash or PowerShell):

```bash
python -m venv venv
```

Activate it (PowerShell: `venv\Scripts\Activate.ps1`; Git Bash: `source venv/Scripts/activate`),
then install PyTorch for your CUDA version first (RTX 50-series: CUDA 13.0):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install -r requirements.lock
```

Optional: `triton-windows` makes `train.compile: true` work on Windows. It is 3-7% faster and
off in the templates.

Every command below assumes the venv is active. Without activating it, write
`venv\Scripts\python.exe` instead of `python`.

---

## 3. Prepare the dataset

Full reference: [datasets.md](datasets.md).

### One folder per person

```
datasets/finetunes/moss/
  prefix.txt                 "photo of [folder]"      <- optional, see below
  people/                    <- dataset.path
    feliciaday/
      001.jpg
      001.txt                "a woman with red hair smiling, close-up, soft window light"
      002.png
      002.txt
    keanureeves/
      ...
  reg/                       <- optional regularisation photos (step 3c)
    women/  ...
    men/    ...
```

- Images: `.jpg`, `.jpeg`, `.png`, `.webp`, at any depth below `dataset.path`.
- **Captions:** a `.txt` with the same name is used **exactly as written**, and the trigger is
  **not** added. So either write the trigger into each caption, or let `prefix.txt` add it:
  `photo of [folder]` puts `photo of feliciaday, ` in front of every caption in the `feliciaday`
  folder.
- **The folder name is the trigger** when you use `[folder]`. Pick names that don't share
  tokens with each other (`annasmith` vs `anna` is fine because longer triggers match first,
  but avoid near-identical spellings).
- **Group shots** (two people in one image) go in a folder of their own, captioned by hand with
  both triggers and who is where.
- **How many images:** 20-30 good, varied images per person worked for likeness (angles,
  lighting, distance). More is fine; RAM and encoding time are the limits.

### 3b. Balance the people

Each image is one sample per epoch, so a person with 60 images trains three times as much as one
with 20. `dataset.folder_repeats` sets a repeat count per folder (fractions allowed):

```yaml
dataset:
  repeats: 1                 # every folder not listed
  folder_repeats:
    emmawatson: 0.5          # already learned well: a random half each epoch
    ben: 3                   # learns slowly: every image three times
```

### 3c. Regularisation photos (optional)

Real photos of the class ("a woman", "a man"), **not model output**, trained at a reduced step
so the class itself doesn't drift. Point `dataset.reg_path` at them; `.txt` captions work as
above. Prior preservation (step 4) needs no extra images and is the first thing to try. The two
can be combined.

### 3d. The encoder cache

The first run VAE-encodes every image (~25 images/s at 512 px) and text-encodes every caption
(~3 captions/s). Results are stored in `.maltrainer_cache/` next to the images, so the next run,
or a resume, only encodes what is new or changed. Moving or renaming folders keeps the cache
valid. Delete a `.maltrainer_cache` folder to clear it.

---

## 4. Write the config

The web UI and `run_moss_image_training.bat` only start LoRA runs, so a finetune config is
written by hand. Copy the template into `config/`, which is git-ignored:

```bash
cp templates/moss_image_finetune_template.yaml config/moss_image_finetune__people.yaml
```

Replace the placeholders:
- `[comfyui_models]`: your ComfyUI `models` folder, with forward slashes
  (`C:/Development/ComfyUI/models`).
- `[person_to_train]`: the dataset folder name, or edit `dataset.path`, `output.*` and the sample
  prompts directly for a multi-person run.

### The settings that matter

| Setting | Template | What to set |
|---|---|---|
| `model.transformer_path` | Base bf16 | keep it. To continue from an earlier finetune's weights (fresh optimizer), point it at that output |
| `dataset.path` | `datasets/[person]` | the folder holding the person folders |
| `dataset.resolution` | 512 | 512 is fast (~0.8 s/step, more with EMA/prior). 1024 is ~3x slower per step; MOSS finished pretraining at 1024 |
| `dataset.trigger` / `default_caption` | the person | **multi-person: set both to `''`**, so an image without a caption trains on nothing instead of the wrong person |
| `finetune.trainable_prefixes` | `["layers."]` | keep it: the 42 blocks. Don't add `text_embedder.` (the trainer warns if you do) |
| `prior.weight` | 0 (off) | **1.0 for multi-person runs.** Keeps "woman" / "man" without a trigger looking like before |
| `prior.classes` | `{person: person}` | every trigger with its class word: `{feliciaday: woman, keanureeves: man}` |
| `train.max_steps` | 1000 | from `--suggest-steps` (step 5) |
| `train.learning_rate`, `automagic_min_lr`, `automagic_max_lr` | all 1e-5 | keep the three equal: that pins the rate. 5e-6 if previews look burned |
| `train.ema_decay` | 0 (off) | **0.999 recommended.** At batch size 1 the raw weights wander; the average is what you keep. Costs 30 GB CPU RAM |
| `train.ema_every` / `ema_warmup` | 4 / false | keep them |
| `train.save_every` | 0 | 0 = save only at the end or on key `s` / `q`. Every save is a 10 GB file |
| `sample.every` | 1000 | 300-500 to watch closely. Each round costs ~24 s per 1024 px image |
| `sample.prompts` | person + class | one prompt per person you want to watch, plus one **without** a trigger to catch leaking |

For a working multi-person example, compare with `config/moss_image_finetune__people.yaml`: 30
people, `folder_repeats`, `prior.classes` for each, a `reg_path`, EMA 0.999, 24,000 steps.

---

## 5. Check it before training

Three checks that load no model and finish in seconds.

**How many steps?**

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml --suggest-steps
```

It lists every folder with its image count, repeats and samples per epoch, and the `max_steps`
that gives each person 22 epochs (the point where the people LoRAs had their likeness). Give
another epoch count to plan for that instead:

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml --suggest-steps 15
```

It also names:
- `folder_repeats` entries that match no folder (a typo would silently do nothing)
- person folders missing from `prior.classes` (they would train without prior preservation)

Copy the number into `train.max_steps` yourself.

**A dry run** counts the epoch from the folders (repeats included) and ticks through every step
with a fake one, so you see the step and epoch totals and the progress bar's pace:

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml --dry-run
```

---

## 6. Train

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml
```

What happens at the start:
1. **Encoding:** new images and captions are encoded, then both encoders are unloaded.
2. **Loading** the bf16 DiT (~4 s) and, with EMA, allocating its CPU copy (~7 s).
3. **Recording the prior** (with `prior.weight > 0`): one forward per image × `samples_per_image`,
   before step 1. It's saved as `<name>_prior.safetensors` and reused on resume.
4. **Step-0 previews:** the base model's own answer to each prompt, the picture everything later
   is compared with.

Then the progress bar runs. A warm start of a small set takes 15-20 s; 10,000 images take ~45 s.

### Keys (in the console window, read at the next step)

| Key | Does |
|---|---|
| `s` | save a checkpoint + resume state now, keep training |
| `q` | save, then stop (ready to resume). If the save fails, training continues instead |
| `p` | pause until `p` again (model stays loaded; `s`, `q`, `h` still work) |
| `e` | preview menu: prompts on/off, interval (←/→ ±100), raw/EMA rows. Training waits while it is open |
| `g` | gallery: 8 trained people at random with generic prompts, to `samples/gallery/` (~3 min; training waits) |
| `h` | list the keys |

You can also **edit the `sample:` section of the config file while training runs**: saving the
file applies it at the next step. Edits outside `sample:` apply at the next start only.

---

## 7. Watch it

### The `[monitor]` line

Every 100 steps:

```
[monitor] step 9200/24000 | loss 0.5638 (low 0.81 n=4, mid 0.52 n=40, high 0.55 n=56) | prior 0.00041 |
signal attn +0.10 mlp +0.08 | nonfinite 0 | drift attn 1.2e-02 mlp 1.1e-02 (t^0.48) | 1.30 s/step | peak 24.9 GB
```

Read it like this (full table: [the reference](moss-image-finetuning.md#watching-a-run-the-monitor-line)):

- **A single step's loss means nothing.** It swings between ~0.25 and 0.9 with the noise level.
  Watch the bands over thousands of steps instead.
- **`prior`** should rise slowly and level off. If it keeps climbing, the people are leaking into
  "woman" / "man": raise `prior.weight`.
- **`drift` with `t^k`:** k near 1 means the weights move in a steady direction (learning). k near
  0.5 with `signal` near 0 means they wander. EMA fixes that; a lower rate doesn't.
- **`nonfinite`** above 0 is logged as a warning. Three NaN losses in a row stop the run
  **without saving**: resume from the last state with a lower rate.

### Previews

They go to `<output.path>/samples/`: `step009000.png` holds all prompts side by side, and
`prompts.txt` names each slot and seed. Every prompt keeps its seed, so a strip compares directly
with `step000000.png`. With EMA you get two rows: `raw` (what is training right now, noisy) and
`ema` (what the checkpoint saves, and the one to judge).

Look for:
- the trigger prompts getting more like each person
- the **trigger-free prompt staying close to step 0.** If it turns into one of your people,
  they're leaking into the class.
- burned, over-sharpened or plastic faces: the rate is too high (try 5e-6)

---

## 8. Stop, resume, change the dataset

**Stop:** press `q`. It saves `<name>_step_<N>.safetensors` and the resume state
`<name>_state/`, then exits. Only the latest state is kept, and it is swapped in whole, so an
interrupted save leaves the previous one intact.

**Resume exactly** (weights, optimizer, EMA, step count, data order, RNG):

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml --resume-from output/moss_image_finetune/people/moss_image_ft_people_state
```

or set `train.resume_from` in the config.

**Train past the old end:** raise `train.max_steps`. It counts from the start of the run, not from
the resume: at step 8505, seven more epochs of 2033 steps is `8505 + 7 × 2033 = 22736`.

**Add people, photos or captions between runs:** that works. On resume the trainer sees that the
dataset changed, logs `The dataset changed since the saved state (482 samples then, 2465 now)`,
and starts a fresh epoch. Weights, optimizer, EMA and the step count carry on. Before resuming:
1. add the new people to `prior.classes`, or they get no prior preservation (`--suggest-steps`
   warns you)
2. keep `model.transformer_path` on the original base, because the prior for new images is
   recorded from it (~10 min for 440 new images × 8 at 512 px)
3. recompute `max_steps` with `--suggest-steps`

**Don't change on resume:** `finetune.trainable_prefixes` and `train.automagic_param_groups`. The
optimizer state no longer fits them and the trainer refuses. To change what trains, start from
the weights instead (point `model.transformer_path` at the checkpoint; the optimizer restarts).

**Keeping an older state:** copy the whole folder (e.g. to `..._state_8505`) and point
`--resume-from` at the copy. It must contain `model.safetensors`, `optimizer.pt` and
`trainer_state.pt` (and `ema.safetensors` with EMA on).

---

## 9. Use the result

Every checkpoint is a full ComfyUI model file with the same key layout as the base:
- **In ComfyUI:** put it in `models/diffusion_models/` and load it like
  `moss_image_base_bf16`. ComfyUI's MOSS template uses res_multistep, simple schedule, 30 steps,
  CFG 4, with SkipLayerGuidance on layer 22 for the first half. With the 8-step DMD LoRA it takes
  8 steps; see [moss-image-comfyui-prompting.md](moss-image-comfyui-prompting.md) for that setup
  and how to prompt the finetune.
- **In maltrainer:** as `model.transformer_path` for LoRAs or further finetuning.
- **As a LoRA** (smaller, stackable with other LoRAs, loads on the base):

  ```bash
  python scripts/extract_lora.py C:/Development/ComfyUI/models/diffusion_models/MOSS/moss_image_base_bf16.safetensors output/moss_image_finetune/people/moss_image_ft_people.safetensors --rank 32 64 128
  ```

  One file per rank. The base must be the exact checkpoint the finetune started from. See
  [extract-lora.md](extract-lora.md) for `--mode` and the other options.

The training config and run details are in each file's safetensors metadata
(`maltrainer_architecture: moss_image_finetune`).

---

## Command cheat sheet

| Do | Command |
|---|---|
| Copy the template | `cp templates/moss_image_finetune_template.yaml config/my_ft.yaml` |
| Steps for 22 epochs per person | `python train_moss_image_finetune.py --config config/my_ft.yaml --suggest-steps` |
| Steps for N epochs | `python train_moss_image_finetune.py --config config/my_ft.yaml --suggest-steps 15` |
| Simulate the schedule | `python train_moss_image_finetune.py --config config/my_ft.yaml --dry-run` |
| Simulate, slower fake steps | `python train_moss_image_finetune.py --config config/my_ft.yaml --dry-run --dry-run-step-time 0.1` |
| Train | `python train_moss_image_finetune.py --config config/my_ft.yaml` |
| Resume | `python train_moss_image_finetune.py --config config/my_ft.yaml --resume-from output/.../<name>_state` |
| Finetune → LoRA | `python scripts/extract_lora.py <base.safetensors> <finetune.safetensors> --rank 64` |
| Read a file's training metadata | `python scripts/inspect_lora_metadata.py <file.safetensors>` |
| Follow the log | `tail -f output/moss_image_finetune/<name>/training.log` |

Keys while training: `s` save · `q` save and stop · `p` pause · `e` preview menu · `g` gallery · `h` help.

---

## Troubleshooting

### Errors that stop the run

| Message | Cause | Fix |
|---|---|---|
| `... is int8; finetuning needs the plain bf16 DiT` | `transformer_path` points at the int8 file | use `moss_image_base_bf16.safetensors` |
| `train.resume_from: .../model.safetensors is missing` | the folder isn't a complete state: a wrong path, a partial copy, or a name that doesn't exist (e.g. `_state_8530` when the save went to `_state`) | point at the folder the log named in `Saved resume state (step N) to ...` |
| `gradient_accumulation_steps must be 1` | fused Automagic3 steps inside backward | set it to 1, or `automagic_fused: false` (+~10 GB VRAM) |
| `prior.weight > 0, but no caption names a trigger of prior.classes` | the captions don't contain the trigger words (e.g. no `prefix.txt` and no trigger in the `.txt` files) | check a few captions, or the `prefix.txt` with `[folder]` |
| `dataset.reg_path ... has no images` | wrong regularisation path | fix the path or set `reg_path: ""` |
| `No usable images: every file in the dataset failed to load` | wrong `dataset.path`, or unreadable files | check the path; the log names each skipped file |
| `MOSS-Image finetuning runs on one GPU` | started with accelerate on several GPUs | run plain `python` on one GPU |
| optimizer groups don't match on resume | `trainable_prefixes` or `automagic_param_groups` changed since the state was saved | change them back, or continue from the weights with a fresh optimizer |
| stops after NaN losses, nothing saved | three non-finite losses in a row | resume from the last state with a lower rate (5e-6) |
| CUDA out of memory | a large bucket at high resolution, or something else on the GPU | lower `dataset.resolution`, close other GPU programs, check `peak` in the monitor line |

### Things that look wrong but aren't

| You see | What it means |
|---|---|
| `The dataset changed since the saved state (...); resuming at step N with a fresh epoch` | expected after adding or removing images or captions. Training carries on |
| `The resumed run is already at step N of N; raise train.max_steps to continue` | the run had finished. Raise `max_steps` |
| The epoch count jumps after a resume with more images (e.g. epoch 19 of 25) | epochs before the change were smaller; the total is recomputed from the steps left. Fixed in `3e2602d`; older runs showed `19/19` |
| Loss jumping between 0.3 and 0.9 every step | normal: each step is one image at one random noise level. Watch the monitor bands |
| The `raw` preview row changes look every few hundred steps | batch-size-1 wandering. Judge the `ema` row |
| A prompt added mid-run renders slightly differently after a restart | it was encoded on the CPU mid-run (~2.6% off the GPU encoding); previews only |
| A save with `q` says the disk is full and training continues | intended: the checkpoint is skipped with the space it needs; free space and press `s` again |
| A seed renders differently in ComfyUI | ComfyUI uses another noise generator and an unsnapped schedule |

### Quality problems

| Problem | Try |
|---|---|
| The trigger-free prompt turns into one of the people | raise `prior.weight` (1.0 → 2.0), or add regularisation photos |
| A person isn't getting their likeness | more repeats for that folder (`folder_repeats: 3`); check their captions name the trigger; more varied images |
| A person learned well early and is now overcooked | lower their repeats (0.5) for the rest of the run |
| Burned, plastic or over-contrasted faces | lower the pinned rate to 5e-6 (all three values) |
| Everything changes look from preview to preview | turn on EMA (0.999, `ema_every: 4`, `ema_warmup: false`) |
| Likeness right, fine detail (skin, hair) soft | try `train.timestep_logit_mean: 0`. The default trains mostly at high noise (only ~2% of steps below σ 0.3) |
| People blending into each other | distinct triggers; group shots captioned with who is where |

---

## Other models

Two models have a finetune trainer: MOSS-Image (every trained weight on the GPU at once) and
Krea 2 ([krea2-finetuning.md](krea2-finetuning.md): rotating windows over an fp8 frozen model,
`maltrainer/finetune/rotation.py`). What they are built from:

- **Shared, model-agnostic** (`maltrainer/finetune/`): parameter selection by name prefix
  (`selection.py`), prior preservation (`prior.py`), regularisation images at a reduced step
  (`regularisation.py`), per-folder repeats (`repeats.py`) and LoRA extraction (`extract.py`).
  `--suggest-steps` uses `maltrainer/utils/step_planner.py`.
- **Rotating windows** (`rotation.py`): for a model whose bf16 weights don't fit beside their
  training state. Any model whose block Linears load as `Fp8ScaledLinear` can use it.
- **MOSS-Image's own** (`maltrainer/models/moss_image/finetune.py`): the default trainable set
  (`layers.`), the Automagic3 parameter groups, and how to save a full checkpoint in the source
  file's exact key layout (including its fused q/k/v).

A finetune trainer for another model needs only that model-specific part, plus a trainer script
modelled on `train_moss_image_finetune.py` that reuses the model's existing LoRA trainer for data,
caching, timesteps and loss. Until one exists, train LoRAs for those models:

| Model | Guide |
|---|---|
| FLUX.3 [klein] | [flux3-klein-lora-training.md](flux3-klein-lora-training.md) |
| Krea 2 | [krea2-lora-training.md](krea2-lora-training.md) |
| Qwen-Image 2.1 | [qwen-image-2.1-lora-training.md](qwen-image-2.1-lora-training.md) |
| Z-Image | [zimage_lora_training.md](zimage_lora_training.md) |
| MiniMax H3 | [minimax-h3-lora-training.md](minimax-h3-lora-training.md) |
| MOSS-Image | [moss-image-lora-training.md](moss-image-lora-training.md) |

The model files differ per model, but the dataset layout, captions and `prefix.txt` in this
guide are shared by the MOSS-Image, Krea 2 and FLUX.3 [klein] trainers
([datasets.md](datasets.md)). The keys (`s`, `q`, `p`, `h`) work in the MOSS-Image and Krea 2
trainers.
