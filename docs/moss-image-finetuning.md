# MOSS-Image full finetuning

`train_moss_image_finetune.py` trains the MOSS-Image DiT's own weights instead of a LoRA. By
default it trains the Linear weights of the 42 joint blocks (4.76B parameters); any subset, up to
all 5.08B, can be selected. Data, caching, timesteps and loss are the LoRA trainer's
([moss-image-lora-training.md](moss-image-lora-training.md)); only the model side differs.

The model-agnostic parts live in `maltrainer/finetune/`: parameter selection, prior preservation,
regularisation images and LoRA extraction. What is MOSS-Image's own (its default trainable set,
optimizer groups, checkpoint layout and fused q/k/v) is in
`maltrainer/models/moss_image/finetune.py`.

```bash
python train_moss_image_finetune.py --config templates/moss_image_finetune_template.yaml
python train_moss_image_finetune.py --config my_run.yaml --resume-from output/.../moss_image_ft_x_state
```

The template has placeholders (`[comfyui_models]`, `[person_to_train]`). Copy it and fill them in;
the web UI and `run_moss_image_training.bat` do not offer finetuning yet.

## How many steps: `--suggest-steps`

```bash
python train_moss_image_finetune.py --config my_run.yaml --suggest-steps        # 22 epochs
python train_moss_image_finetune.py --config my_run.yaml --suggest-steps 15     # 15 epochs
```

It counts the images without loading a model and exits:
- images per subject folder and per regularisation folder
- one epoch's size: every subject `repeats` times plus every regularisation image `reg_repeats`
  times, divided by `gradient_accumulation_steps`
- the `train.max_steps` that gives each person that many epochs, and how many epochs the config
  gives now

With prior preservation on, it also names any subject folder missing from `prior.classes`: those
people would train without it. Copy the suggested number into the config yourself; the default of
22 epochs is where the people LoRAs had their likeness.

## Watching a run: the `[monitor]` line

A single step's loss says little: it is one image at one noise level, and swings between about
0.25 and 0.9 with the sigma drawn. Every `train.monitor_every` steps (default 100) the console and
`training.log` get a summary instead:

```
[monitor] step 400/8932 | loss 0.5821 (low 0.214 n=13, mid 0.512 n=41, high 0.731 n=46) | prior 0.00041 |
signal attn +0.12 mlp +0.09 | nonfinite 0 | drift attn 1.9e-03 mlp 2.4e-03 (t^0.92) | 1.31 s/step | peak 21.4 GB
```

| Field | What it is | Healthy | Trouble |
|---|---|---|---|
| `loss` (bands) | the window's mean loss, overall and by noise level: low σ < 0.5, mid < 0.85, high | each band steady or slowly falling against its own earlier values. The bands differ by nature: with the velocity target, `low` is the highest (~0.8) | a band rising steadily over thousands of steps |
| `prior` | the prior-preservation term's mean | near 0 at first, rises slowly, then levels off | keeps climbing while likeness improves: the people leak into "woman" / "man", so raise `prior.weight` |
| `signal` | Automagic3's direction vote per group: the share of weights whose updates kept one sign over 8 steps, minus the share that flipped every step | above 0 | ~0 together with drift growing like t^0.5: the weights wander more than they learn (see below). Clearly below 0: overshooting, so lower the rate |
| `nonfinite` | NaN / inf gradient values since the last line; the optimizer zeroes them, so only this count shows them | 0 | anything above 0 is logged as a warning; a few now and then is survivable, a steady stream is not |
| `drift` | \|W − W0\| / \|W0\| of the attention output and feed-forward down projection of blocks 0, 10, 21, 31, 41; W0 = the weights when this process started | keeps growing. At 1e-5 it reached 7.5e-3 in 160 steps (the failed 2.5e-7 run: 5e-3 in 3000) | flat: nothing trains. Shooting up with burned previews: lower the rate |
| `s/step`, `peak` | speed and peak VRAM of the window | steady | a jump in peak can mean a bigger bucket; check for OOM |

`(t^k)` compares the drift with the steps since the start: drift ∝ steps^k. Updates that keep a
direction add up linearly (k ≈ 1); step-to-step noise only grows as a random walk (k ≈ 0.5).

**Signal ~0 with k ~0.5 means the weights are wandering.** The 160-step people test at 1e-5
(2026-10-18, no EMA) read exactly that: signal 0.00, k 0.41-0.49, drift 7.5e-3. Its previews
showed what it means: with the same seeds, every 50 steps brought a different look (plastic and
smooth at 50, photographic at 100, older faces at 150) and a different trigger-free woman, and
the base model's step-0 pictures were the best. At batch size 1 each step's direction comes from
one image at one noise level, so most of each step is noise, and at 1e-5 the noise is big enough
to move the whole model's look. A lower rate would not fix this: it shrinks the wandering and the
learning alike. Averaging does: turn on EMA (below), which keeps the consistent part and cancels
the wandering, or use a larger batch. The drift line keeps ~350 MB of bf16 copies on the GPU;
`train.monitor_drift: false` turns it off.

`s/step` leaves out the time spent rendering previews.

**A NaN loss stops the run.** Three NaN / inf losses in a row raise an error *without saving*, so
the last checkpoint and resume state stay as they were. Resume from the state with a lower rate.

## Previews during training

The `sample:` section renders fixed prompts with the weights in training, every `every` steps and
(with `at_start`) once before the first step, the base model's own answer to compare with.
Training pauses for the render and carries on; nothing is saved but the pictures.

```yaml
sample:
  every: 1000                # 0 = off
  at_start: true
  prompts:
    - "photo of feliciaday, a woman smiling slightly, close-up with face visible"
    - "photo of a woman smiling slightly, close-up with face visible"   # trigger-free: the class
  negative: ""
  steps: 30
  cfg: 3.0
  width: 1024
  height: 1024
  seed: 42                   # prompt i uses seed + i, the same at every step
```

- **Files** go to `<output.path>/samples/`: `step001000_00.png` per prompt, `step001000.png` with
  all of them side by side, and `prompts.txt` naming each index with its seed.
- **Define prompts upfront, switch them while training runs.** Every prompt in `sample.prompts`
  is encoded at the start, enabled or not, so turning one on later is instant. A plain line is
  enabled; this form defines one that is off:

  ```yaml
  prompts:
    - "RAW photo of feliciaday, a woman smiling slightly ..."     # on
    - prompt: "RAW photo of a woman smiling slightly ..."
      enabled: false                                             # off, ready to switch on
  rows: [raw, ema]       # with EMA: which weights to render; [] = none
  ```

- **The preview menu, key `e`:** training waits while it is open.

  ```
  Preview settings   ↑↓ move · space toggle · ←→ every ±100 · Enter apply · Esc cancel
  > Every:   300 steps   (0 = off)
    [x] raw weights
    [x] EMA average
    [x] 00  RAW photo of feliciaday, a woman smiling slightly, close-up with face …
    [ ] 01  RAW photo of a woman smiling slightly, close-up with face visible, sof…
  ```

  ←/→ change the interval by 100 wherever the cursor is (0 = off); space switches the row or
  prompt under the cursor; both rows off means no previews. Enter applies and logs the changes,
  Esc leaves everything as it was. The changes last for the run; the config file is not
  rewritten (that would drop its comments), so set `enabled:` / `every:` / `rows:` there to keep
  them for the next start.
- **Or edit the file while training runs:** change the `sample:` section of the config file you
  started with and save it. At the next step the trainer re-reads that section and logs what
  changed; for the prompts it names, the file wins over earlier menu choices:
  - `every`, `steps`, `cfg`, size, `negative`, `seed` and `at_start` apply from the next round
  - a **new prompt** is encoded on the CPU in the background (~6 s; the GPU is busy training and
    the text encoder does not fit beside it) and joins the next round once it is ready
  - a **removed prompt** stops rendering
  - each prompt keeps the number it was first given (its seed is `seed` + that number, and its
    files are `_NN`), so adding or removing prompts never renumbers the others; `prompts.txt`
    lists every number, marking removed prompts
  - a half-saved or broken file is reported and the previous settings stay
  - edits outside `sample:` are reported and apply at the next start only

  The CPU encoding differs ~2.6% from the GPU's (the int8 layers run a float fallback on the
  CPU), so a prompt added mid-run can render slightly differently after a restart, when it is
  encoded on the GPU. These embeddings serve previews only and never enter the caption cache.
- **The sampler** is the reference pipeline's (`maltrainer/models/moss_image/sampling.py`): the
  shift-5 schedule snapped to the 1000 training sigmas, Euler, CFG against the negative with block
  22 skipped. The VAE decoder is the full FLUX.2 one from `model.vae_path`; MOSS-Image's own small
  decoder avoids some spots in ComfyUI but decodes the same latents.
- **Cost:** about 24 s per 1024 px image at 30 steps with CFG (two DiT passes a step) on the 5090,
  12 s at 768 px. Five prompts every 1000 steps add about 2 minutes per 1000 steps. The prompts are
  encoded with the training captions, so changing them means the text encoder loads once at the
  next start.
- **Comparing:** each prompt keeps its seed, so `step000000.png` against a later strip shows
  exactly what training changed. The trigger-free prompt should stay close to step 0; if it turns
  into one of the subjects, the people are leaking into the class (raise `prior.weight`).
- **ComfyUI renders a seed slightly differently:** its scheduler is not snapped to the training
  grid, and its noise comes from a different generator.

**With EMA on, a fine-tune previews two rows:** the raw weights (`..._raw.png`, what is training
right now) and the EMA average (`..._ema.png`, what checkpoints save). The average is swapped into
the model for its row and the raw weights restored bit for bit afterwards, through the EMA's
pinned staging buffers, so it needs no extra memory (two transfers of the model over PCIe). The
raw row shows the batch-1 wandering; the EMA row should change slowly and steadily. The LoRA
trainer previews the EMA average alone, since that is what its checkpoints save.

## How it fits in 32 GB

- **bf16 weights only.** It needs the plain `moss_image_base_bf16.safetensors`; the int8 file is
  rejected. There is no fp32 master copy: Automagic3 computes each update in fp32 and rounds it
  stochastically into the bf16 weight, so small updates are not lost.
- **Fused Automagic3** (`train.automagic_fused: true`, the default). Each parameter steps inside
  backward as soon as its gradient exists, then the gradient is freed. No full set of gradients
  (10 GB) is ever held.
- **Small optimizer state.** Automagic3 keeps an 8-step sign history (1 byte per parameter) and a
  factored second moment (rows + columns), about 5 GB for the whole model.
- **Gradient checkpointing** per block, as for LoRAs.

Measured on an RTX 5090 (32 GB), all parameters trained:

| | Peak VRAM | Step |
|---|---|---|
| 512 px (trainer, aneta) | 22.2 GB (nvidia-smi, includes caching) | 0.78 s |
| 1024 px (probe) | 16.8 GB (allocated) | 2.26 s |
| 1536 px (probe) | 19.0 GB (allocated) | 6.47 s |

Fused training has two limits:
- `gradient_accumulation_steps` must be 1. Set `automagic_fused: false` to accumulate. That
  holds the gradients, about 10 GB more, so ~26 GB at 1024 px.
- There is no global gradient-norm clip, so `max_grad_norm` is ignored here. Automagic3 clips
  each tensor's update by RMS instead (`automagic_clip_threshold`).

## Learning rate: pinned at 1e-5

The templates pin Automagic3's rate: `learning_rate`, `automagic_min_lr` and `automagic_max_lr`
are all 1e-5, so its sign vote cannot move it, and the LR column in `training.log` stays at 1e-5.

Left to adapt, the rate only falls on MOSS-Image's flat loss. The first people run (3000 steps,
2026-10-18) launched at 1e-6 and ended near 2.5e-7. The block weights are about 0.03 in size,
where bf16 values are 1.2e-4 apart, so a 2.5e-7 step is 0.002 of one spacing. Stochastic
rounding keeps such a step unbiased, but each weight then moves one whole spacing about once in
500 steps, at random. The run's change came out as that noise: every layer's singular values
were flat (the largest 2-3x the median), and LoRAs extracted from it kept under 15% of the
change even at rank 256. Fizgig found the same: its Adaptive LR "walks to its floor" on
MOSS-Image, so its presets use fixed rates.

1e-5 is Fizgig's finetune rate. Its finetune uses Adafactor, whose update Automagic3 shares (a
factored second moment, clipped to RMS 1, no momentum), so the number carries over. On these
weights it is 0.08 of a bf16 spacing per step: still below one spacing, so it relies on
stochastic rounding, but 40x the signal of the failed run. 1e-4 is a LoRA rate; on a full model
it is the top of the range. If previews look burned, try 5e-6 with the same pin, rather than an
adaptive run.

Only Automagic3 is supported. AdamW 8-bit would need 10 GB of gradients plus 10 GB of state. It
also has no stochastic rounding, so on bf16 weights its small updates would round away.

## Startup time

The encoder cache ([datasets.md](datasets.md#encoder-cache-moss-image)) removes re-encoding.
What remains:

| Part | Cost | 10,000 images |
|---|---|---|
| Python + library imports | ~5 s | ~5 s |
| Hash every image (cache keys) | ~2 ms per image | ~20 s |
| Read cached latents + caption embeddings | ~0.4 ms per file | ~8 s |
| Load the bf16 DiT (10 GB) | ~4 s | ~4 s |
| New start: build the optimizer | ~1 s | ~1 s |
| EMA on: allocate the 20 GB CPU shadow | ~7 s | ~7 s |
| Resume: optimizer (~5 GB) + EMA (20 GB) | not measured at full size; read at disk speed | ~10-30 s (est.) |

So a warm start is about 15-20 s for a small set and about 45 s for 10,000 images. A resume adds
the state files, so about 1 minute. A cold cache adds the encoding: ~25 images/s and ~3
captions/s. The measured rows come from aneta/anetalarge on the 5090 workstation (hashing:
270 images, 237 MB, 0.55 s).

## Per-folder repeats

`dataset.repeats` sets how often each image appears in an epoch; `dataset.folder_repeats` sets it
per folder, by its path under `dataset.path` or by the folder's own name:

```yaml
dataset:
  repeats: 1                 # every folder not listed
  folder_repeats:
    emmawatson: 0.5          # trained well: a random half of her images each epoch
    gosiamarek: 3            # trained worst: every image three times per epoch
    people/zendaya: 3        # new: catch up
```

- **Fractional values** work: 2.5 shows every image twice plus a random half once more; 0.5 a
  random half. The half is drawn again every epoch, so all images are used over a run.
- **What a person learns** depends on how often each image is trained on, not on epochs: 3x for 7
  epochs is 21 times per image, as 1x for 21 epochs.
- **Folder size counts too:** a person with 62 images at 1x gets more steps than one with 19
  images at 3x. Set the repeats with that in mind.
- `--suggest-steps` lists every folder with its repeats and samples per epoch, and names any
  `folder_repeats` entry that matches no folder (a typo would otherwise do nothing).
- Regularisation images use `dataset.reg_repeats`, not this table.
- `max_steps` counts from the start of the run: resuming at step 8505 for 7 more epochs of 2033
  steps is `8505 + 7 x 2033 = 22736`.

## What trains

`finetune.trainable_prefixes` lists parameter-name prefixes. Without the key, the default is
`["layers."]`; an explicit `[]` trains everything. `finetune.linears_only` (default `true`) then
keeps only Linear weights: norm scales, biases and other vectors stay frozen.

| Prefixes | `linears_only` | Parameters |
|---|---|---|
| `["layers."]` (default) | `true` (default) | the 42 blocks' Linears, 4.76B |
| `["layers."]` | `false` | the 42 blocks with their norms |
| `[]` | `false` | all 5.08B |
| `["layers.40.", "layers.41."]` | `true` | two blocks, 0.23B |
| `["text_embedder."]` | `true` | the 2 text-embedder blocks and the text projection (logs a warning) |

Why the default leaves out the text side, as OpenMOSS's guide and Fizgig do:

- **The text embedder** (a projection and 2 blocks) refines every prompt before it meets the
  image. Retraining it on a few hundred captions changes how all prompts are read, not only the
  trigger words.
- **Norms, biases and the shared modulation** are few parameters, but each one moves a whole
  channel of every token. They drift the model's overall look much faster than a matrix entry.
- **Fewer optimizer groups.** Only `attn` and `mlp` remain, so Automagic3's rates are not pulled
  around by the tiny `other` / `norm` groups.

Automagic3 adapts one rate per family: `attn` (q/k/v/o, and the text blocks' qkv/proj), `mlp`,
`norm` (the blocks' RMSNorm weights) and `other` (embedders, modulation, output head).

A state saved by a run with a different trainable set cannot resume: its optimizer groups no
longer match, and the trainer says so. Continue from its weights instead (point
`model.transformer_path` at the checkpoint).

## Prior preservation (DOP)

A full finetune has no low-rank bound. Over a long run on a few people it also changes what the
model makes for "woman" or "man" without a trigger. Prior preservation holds the trigger-free
model to what it did before. It is AI Toolkit's Differential Output Preservation, applied to
finetunes the way Fizgig does it:

1. **Before step 1**, for each training image whose caption names a trigger, the trainer writes
   a class caption with each trigger replaced by its class word ("photo of feliciaday, a woman
   with short hair ..." becomes "photo of woman, a woman with short hair ..."). It records the
   original model's prediction for `samples_per_image` noised versions of the image with that
   caption. The sigmas come from the training sampler.
2. **Each step** on such an image adds `prior.weight` × the MSE between the model's prediction
   now and the recorded one, for one of those inputs. Both terms go through one backward, so
   fused Automagic3 still steps once.

```yaml
prior:
  weight: 1.0                # 0 = off
  classes: {feliciaday: woman, gilliananderson: woman}   # trigger: class word
  samples_per_image: 8
  every: 1                   # add the term every N steps
```

With a single `dataset.trigger`, `prior.class_word: woman` works instead of `classes`. Triggers
match whole words, any case; longer ones go first, so `anetahoppe` wins over `aneta`. Images
whose caption names no trigger get no prior term.

What it costs:

- **Recording:** one forward per recorded input before training starts (8 × 138 images at 512 px
  is about 1,100 forwards).
- **Storage:** each recorded prediction is kept in bf16 on the CPU (0.26 MB at 512 px); only the
  sigma and a noise seed are stored for the input. The record is saved as
  `<name>_prior.safetensors` in the output folder and hard-linked into each resume state, since a
  resumed model is no longer the original.
- **People added later:** a resumed run whose dataset has images the saved prior does not cover
  (new people, new photos of old ones) records theirs from the original model too. It puts the
  base checkpoint's values (`model.transformer_path`) back into the trained parameters for the
  recording and restores the trained ones bit for bit afterwards; the frozen parameters never
  changed, so the model is exactly the original meanwhile. About 6 inputs a second at 512 px
  (440 new images x 8 inputs: ~10 min), and ~9.5 GB of CPU RAM for the trained values meanwhile.
  Add the new people to `prior.classes` first, or their captions name no trigger and they get no
  prior (`--suggest-steps` warns). `model.transformer_path` must still be the run's original base.
- **Training:** a second forward and backward on every step it runs, so roughly double the step
  time at `every: 1`. `every: 2` halves that.

The `Prior:` column in `training.log` is the term's mean over the step. It starts near 0 and rises
as training moves the model. If it climbs steadily, the subjects are leaking into the class.

## Regularisation images

Ordinary photos of the broader class, trained beside the subjects at a reduced step size, anchor
the model's idea of the class (Fizgig's "Regularisation images").

```yaml
dataset:
  reg_path: "datasets/reg/women"   # real photos of the class, with .txt captions
  reg_default_caption: "a photo of a woman"   # for images without a .txt
  reg_repeats: 1
train:
  reg_lr_multiplier: 0.2
```

- **The step size.** `reg_lr_multiplier` scales the optimizer's step, not the loss. Automagic3
  divides each update by its own RMS, so a scaled loss would barely change the step. 0.1-0.3
  anchors the class while the subjects train; toward 1.0 the folder trains like a second subject
  set.
- **Use real photos, not the model's output.** Anchoring a finetune to its own samples distils
  its artifacts back in. Caption them normally: anything left unsaid gets attributed to the
  class word.
- **Epochs.** Each epoch contains every subject image `repeats` times and every regularisation
  image `reg_repeats` times, shuffled together, so they count toward the step budget.
- **Resume.** Adding or removing a regularisation folder changes the dataset fingerprint. The
  run then continues with a fresh epoch, as for any dataset change.

Prior preservation needs no extra data and is the first thing to try. The two combine.

## Turning a finetune into a LoRA

[scripts/extract_lora.py](extract-lora.md) takes the difference between a finetune and its base
and keeps each layer's strongest directions with an SVD:

```bash
python scripts/extract_lora.py C:/.../moss_image_base_bf16.safetensors output/moss_image_finetune/people/moss_image_ft_people.safetensors --rank 32 64 128
```

The result is an ordinary MOSS-Image LoRA (q/k/v split as ComfyUI expects) that loads on the
base model and stacks with other LoRAs.

## torch.compile

`train.compile: true` compiles each block, as for LoRAs (see
[moss-image-lora-training.md](moss-image-lora-training.md#torchcompile)). The fused optimizer
still steps inside the compiled backward.

## EMA in CPU RAM

`train.ema_decay > 0` keeps an fp32 average of every trained weight in CPU RAM: 20 GB for the
whole model. Checkpoints then save the average. Each update copies the weights into a pinned
buffer (10 GB over PCIe). A background thread then averages them in 1M-element chunks while
training continues. The next update waits for that thread.

Measured cost at 512 px, all parameters (9800X3D, 8 threads; the average takes ~1.1 s and is
memory-bound): 0.78 s/step without EMA, 1.61 s/step with `ema_every: 1`. At 1024 px a step takes
longer than the average, so only the ~0.4 s copy remains. `ema_every: N` updates every N steps
with the decay raised to the N-th power, which keeps the same time constant at 1/N of the cost.
The template uses 4.

Use a long average for a finetune (0.999, a ~1000-step time constant), not the LoRA templates'
0.98. The people config turns it on for that reason: at batch size 1 the raw weights wander (see
the monitor section), and the average is what keeps the learning without the wandering.

**Turn the warmup off for a finetune** (`ema_warmup: false`). The warmup starts the average short
and lengthens it with the number of updates: it spans about (updates / 9) steps. With an update
every 4 steps that is ~18 steps at step 600 and ~250 at step 8932, never the 1000 that 0.999
means, so it barely averages out the wandering (seen in the first 600-step people test: its EMA
row wandered almost as much as the raw one). Without the warmup the average spans ~1000 steps
from the start. It begins as the base weights and leans toward them early on; that pull is 55% at
step 600, 5% at step 3000, and gone by the end of a long run.

## Output

Every checkpoint is a full ComfyUI single file, about 10 GB. It has the source file's exact key
layout: the same keys, order and dtypes, with `rope.bands` and any `comfy_attention.config`
copied through. ComfyUI loads it like `moss_image_base_bf16`, and so does this trainer, as a
`transformer_path` for LoRAs or further finetuning. The training config and run details are in
the safetensors metadata, as for LoRAs (`maltrainer_architecture: moss_image_finetune`).

The templates use `save_every: 0`: the run saves only at the end, or when you press `s` (save,
continue) or `q` (save, stop). Each save adds a 10 GB model. The resume state is replaced, not
added. Periodic saves still work (`save_every: 500`), but 1000 steps at `save_every: 250` write
40 GB.

Without EMA, the state's `model.safetensors` is a **hard link** to the checkpoint saved with it:
one file on disk under two names. The state then costs only the optimizer (~5 GB), so a run
that ends with one final save uses about 15 GB. Deleting the checkpoint later is safe: the state
keeps its link to the data. With EMA, the checkpoint holds the average and the state needs the
raw weights, so it writes its own copy (10 GB + 20 GB EMA).

## Keys while training

In a console window, at the next step:
- `s` saves a checkpoint now (`<name>_step_<N>.safetensors`, plus the resume state) and training
  continues.
- `q` does the same, then stops, ready for `--resume-from <name>_state`. No `final` file is
  written. If the save fails (e.g. a full disk), training continues instead of stopping.
- `p` pauses training until `p` is pressed again: the GPU sits idle with the model loaded, e.g.
  while you free disk space or check the previews. `s`, `q` and `h` still work while paused. The
  pause is left out of the monitor's s/step.
- `e` opens the preview menu: prompts on/off, the interval, the raw/EMA rows (see
  [Previews during training](#previews-during-training)). Training waits while it is open.
- `h` lists the keys.

Each save of a full model takes ~10-20 s, with 10 GB per checkpoint.

## Continuing

There are two ways:

- **Continue from weights.** Point `model.transformer_path` at a finetuned output. The optimizer
  starts fresh: Automagic3's rate restarts at `learning_rate` and has to find its level again.
- **Resume exactly.** Each save also writes `<name>_state/`, unless `train.save_state` is false.
  Pass that folder to `train.resume_from` or `--resume-from`. It continues with the same weights,
  optimizer state (per-group rates, sign history, second moments), EMA, step count, position in
  the epoch's shuffled order, and Python/torch/CUDA RNG state. To train past the old end, raise
  `max_steps`. Keep `finetune.trainable_prefixes` and `automagic_param_groups` unchanged; the
  trainer checks the optimizer groups.

**Changing the dataset on resume** works: add, remove or recaption images. Weights, optimizer,
EMA and the step count carry on. The position in the old epoch's shuffled order does not, since
it means nothing for a different set. The state records a fingerprint of the samples (paths
relative to the dataset folder, captions, `repeats`). When it differs, the run logs a warning and
starts a fresh epoch. Training continues until `max_steps`, however many epochs the new set
needs. A moved dataset folder still counts as unchanged. Automagic3 keeps its learned rates,
which were tuned to the old data; it re-adapts within its 8-step window.

Tested 2026-10-18 on two trained blocks:
- 10 images for 13 steps, then resumed to 16 on the same data: it continued at "epoch 2, sample 3".
- Grown to 23 images and resumed to 40: a fresh epoch from step 16, through to step 40. Only the
  13 new images were encoded.

The state holds the raw weights (10 GB, hard-linked to the checkpoint when EMA is off), the
optimizer (~5 GB for everything) and the EMA (20 GB if on). Only the latest state is kept. It is written to `<name>_state.tmp` and swapped in whole,
so an interrupted save leaves the previous state intact.

Verified (2026-10-18, two blocks trained): a 3-step run resumed to step 6 matched an
uninterrupted 6-step run. Each step drew the same sample and sigma, losses agreed to ~4 decimal
places, and weights to within one bf16 step. Two fresh runs differ by as much, from GPU
nondeterminism.
