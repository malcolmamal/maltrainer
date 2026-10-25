# Krea 2 full finetuning

`train_krea2_finetune.py` trains the Krea 2 DiT's own weights instead of a LoRA, on one 32 GB
card. Captions, timesteps and loss are the LoRA trainer's
([krea2-lora-training.md](krea2-lora-training.md)). Regularisation images, per-folder repeats,
prior preservation, previews, the gallery and `--suggest-steps` work as in the MOSS-Image finetune
([moss-image-finetuning.md](moss-image-finetuning.md)), from the same code
(`maltrainer/trainers/finetune/`: `dataset.py`, `prior_setup.py`). The walkthrough for finetunes in general is
[finetune-tutorial.md](finetune-tutorial.md).

```bash
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml --suggest-steps
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml --dry-run
python train_krea2_finetune.py --config config/krea2_finetune__people.yaml --resume-from output/krea2_finetune/people/krea2_ft_people_state
```

Start from `templates/krea2_finetune_template.yaml`.

> **Status (2026-10-25):** run on an RTX 5090: a people finetune of 3, then 8 people to 10,000 steps, resumed
> mid-way with new people added, and a one-person run on a merged base. The measurements and the likeness curves are
> in [Results](#results-two-runs-on-an-rtx-5090). The CPU tests cover the mechanics: windows rotate in order, frozen
> weights stay bit-identical, a resumed run ends bit-identical to an uninterrupted one (with the prior and
> fractional repeats too), saved files load back. The preview decoder was checked on the real
> `qwen_image_vae.safetensors`: a 384 px photo encoded and decoded comes back at 39 dB PSNR.

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

## Memory, speed and disk

Measured on an RTX 5090 (32 GB) at 512 px, prior preservation on (`every: 1`):

| | Measured |
|---|---|
| Peak VRAM (the monitor's `peak`, attention window, with the prior's second forward pass) | **29.3 GB** |
| Step | ~2.0 s (one person) to ~2.6 s (eight people, more aspect buckets) |
| Window swap (write back, re-quantize, bring up the next) | 1-7 s, every `rotate_every` steps |
| Recording the prior (8 noised inputs per image) | 2.4 s per image |
| Preview, 768 x 768, 28 steps, CFG 4.5 | ~29 s per image |
| Preview, 1072 x 1472, 10 steps, CFG 1 | ~13 s per image |
| Save: the bf16 model / the resume state (weights hard-linked) | 22 s / 13 s |

That peak leaves little room: **the GPU must be free.** With ComfyUI holding ~18 GB beside it, the run did not fail,
it slowed from ~2.5 s to 53 s a step: on Windows the driver moves what does not fit into system RAM instead of
raising an out-of-memory error. Close other GPU programs (`nvidia-smi` should show a few GB in use), and set
NVIDIA Control Panel -> Manage 3D settings -> Program Settings -> `python.exe` -> **CUDA - Sysmem Fallback Policy:
Prefer No Sysmem Fallback**, so running out of VRAM is an error you see rather than a run 20x slower.

CPU RAM: the master (25.8 GB) plus every frozen window's optimizer state (~12.5 GB), ~40 GB; plus ~8 GB while
Qwen3-VL encodes a prompt added mid-run on the CPU.

Disk: each checkpoint is a full bf16 model (25.6 GB). The resume state hard-links the latest one and adds the
optimizer states, so `save_every` costs 26 GB per save.

## The base model

- **A bf16 file is best**, e.g. Comfy-Org's `krea2_raw_bf16.safetensors` (~26 GB) in
  `ComfyUI/models/diffusion_models/Krea2/`, or a bf16 model you merged
  ([merge-lora-into-model.md](merge-lora-into-model.md)). Its 224 block Linears are quantized to fp8 on load for the
  frozen trunk; the master copy keeps the exact bf16 weights.
- **`krea2_raw_fp8_scaled` works**, with two warnings: its blocks start from dequantized fp8 weights (every saved
  file keeps fp8's rounding), and its 32 text-fusion Linears are fp8 too, so they are dequantized into trainable bf16
  at load. The people runs below trained from it.
- **Extracting a LoRA or adding the finetune to another model needs the exact base it started from, as a plain
  file.** For a run from `krea2_raw_fp8_scaled` that is the fp8 file dequantized to bf16 exactly as the loader reads
  it; the people run wrote it next to its output as `output/krea2_finetune/<run>/_base_dequantized_bf16.safetensors`
  (25.6 GB; delete it with the run, not before you have extracted or merged).
- **Train RAW, not Turbo:** Turbo is distilled (CFG-free, fixed shift) and the training loss is the RAW model's.
  Merged bases are fine: a 50/50 blend of Kroma 0.3.1 Turbo and Krea2Realism (plus LoRAs) trains like any other
  base (see [Rendering](#rendering-a-krea-2-finetune) for how to preview it).

## Optimizer and rate

Fused Automagic3 only: each weight steps inside backward as soon as its gradient exists, so no
full gradient set is held, and the update is rounded stochastically into the bf16 weight, so small
steps are not lost. `gradient_accumulation_steps` must be 1, and there is no global gradient clip
(`automagic_clip_threshold` clips each tensor's update instead).

The template pins the rate at **1e-5** (`learning_rate` = `automagic_min_lr` = `automagic_max_lr`),
Fizgig's Krea 2 finetune rate. Fizgig: 1e-4 "trains, but treat it as the top of the experiment
range"; when a run looks almost right but overcooked, lower the rate rather than the step count.

## How many steps: `--suggest-steps`

`--suggest-steps [EPOCHS]` counts the images per subject folder and in the regularisation folder,
with their repeats, and prints a `train.max_steps` for that many epochs (default 22), then exits
without loading a model. With prior preservation on, it also names subject folders that
`prior.classes` misses. The window cycle is `4 x rotate_every` steps; round the suggestion to a
multiple of it so every window trains equally.

## Per-folder repeats and regularisation images

```yaml
dataset:
  repeats: 1
  folder_repeats: {"people/bob": 3, clara: 0.5}   # by path under dataset.path, or the folder's name
  reg_path: "datasets/reg_people"   # ordinary photos of the class, not model output
  reg_default_caption: ""
  reg_repeats: 1
train:
  reg_lr_multiplier: 0.2
```

Both are the MOSS-Image finetune's (its doc has the details): a fractional repeat shows a random
part of the folder each epoch, and a regularisation image trains at `reg_lr_multiplier` times the
step (Automagic3's `lr_scale`, set on the active window's optimizer and the text fusion's before the
fused backward), not at a smaller loss. Adding or removing images, folders or repeats changes the
dataset fingerprint: a resume then starts a fresh epoch.

## Prior preservation (DOP)

```yaml
prior:
  weight: 1.0
  classes: {anne: woman, bob: man}   # trigger: class word
  samples_per_image: 8
  every: 1
```

As in the MOSS-Image finetune: before step 1 the original model's predictions are recorded for
each image whose caption names a trigger, with the trigger replaced by its class word
("photo of anne" -> "photo of woman"), on `samples_per_image` noised inputs (their sigmas
from the training sampler, at the image's own token shift). Each step on such an image then adds
`weight` x the MSE to one of them, in the same fused backward. `train.dop` (the LoRA trainer's) is
refused; use `prior:`.

- **The original model is the fp8 trunk** the frozen windows run on, with the bf16 text fusion.
  During training the active window is bf16, so the term starts a little above 0 (that window's
  fp8 rounding), not at 0.
- The predictions go to `<name>_prior.safetensors` (0.13 MB per input at 512 px) and are
  hard-linked into each resume state. A resume reloads them; a resume asking for a prior whose run
  had none is refused (the original model is gone).
- **Images added before a resume** (new people) are recorded from the original weights:
  `finetune/rotation_base.py` re-quantizes every rotating weight from `model.transformer_path`
  (the run's original base, which must still be there) and swaps the text fusion back for the
  recording, then re-quantizes from the trained master, bit for bit. It reads one tensor at a time.

## Previews and the gallery

```yaml
sample:
  every: 500
  at_start: true
  prompts: ["photo of anne smiling", {prompt: "photo of a woman smiling", enabled: true}]
  negative: ""
  steps: 28
  cfg: 4.5
  width: 1024
  height: 1024
  seed: 42
  gallery: {count: 8}
```

Rendered with the weights in training, before step 1 (the base model, what later previews compare
with) and every `every` steps, into `<output.path>/samples/`. The sampler
(`maltrainer/models/krea2/sampling.py`) is Fizgig's for the RAW model: Euler on the velocity,
the schedule shifted by the image's token count as in training, CFG against the negative, seeded
fp32 noise from the CPU (a seed renders the same picture every time; prompt slot N uses seed + N). 28 steps at CFG
4.5 are 56 DiT passes per image, ~29 s at 768 px. The decoder is the Wan 2.1 VAE's
(`maltrainer/models/krea2/vae_decoder.py`), from the same file as the encoder.

`width` and `height` must be multiples of 16 (VAE 8x, patch 2). A size off that grid stops the run when it starts,
before any caching, with the nearest sizes that work ("use 1072 or 1088 for 1080"); one saved into the config
mid-run is logged and the last good size kept.

The rest is the MOSS-Image finetune's: key `e` opens the preview settings, saving the config file
applies its `sample:` section without stopping (prompts added there are encoded by Qwen3-VL-4B on
the CPU, ~8 GB of RAM while it is loaded), and key `g` renders the gallery of the people in
`prior.classes`. A finetune here has no EMA, so there is one row: the raw weights.

## Output and resume

- **Checkpoints** are full bf16 ComfyUI files at `<output.path>/<name>.safetensors` (and `<name>_step_N` for
  `save_every` and `s`), with the source's key prefix: load them in ComfyUI like `krea2_raw_bf16` (a hard link
  into `ComfyUI/models/diffusion_models/Krea2/` costs no space), or as `model.transformer_path` here (for LoRAs or
  more finetuning). `scripts/convert_int8_convrot.py` makes an INT8 ConvRot copy (13.3 GB) for ComfyUI.
- **Keys:** `s` save and continue, `q` save and stop, `p` pause, `e` preview settings, `g` gallery
  (with people in `prior.classes`), `h` help.
- **Resume** with `--resume-from <name>_state`: the weights, every window's optimizer state, the
  step (so the window), the position in the epoch and the RNG. Verified bit-exact on the CPU test.
  Raise `max_steps` to train past the old end. Keep `finetune.windows` as it was; the trainer
  refuses a different set.
- **A changed dataset** on resume starts a fresh epoch; the rest carries on.
- **As a LoRA:** `python scripts/extract_lora.py <its plain base>.safetensors <finetune>.safetensors --rank 256 128 64`
  ([extract-lora.md](extract-lora.md); the base must be the one it trained from, see above).
- **Control API:** `control_api: {enabled: true, port: 8765}` publishes the run to sd-backend's MalTrainer tab
  ([control-api.md](control-api.md)); the rate shown is the active window's.

## Results: two runs on an RTX 5090

People here are named Anne, Bella, Clara, Dana, Ella, Felicia, Greta and Hanna; the triggers were their folder
names (`people/<name>/`, captioned "photo of [folder], a woman ..." through `people/prefix.txt`). Base:
`krea2_raw_fp8_scaled`, 512 px, rate 1e-5 pinned, prior on (`weight: 1`, class "woman"), regularisation photos of
women at about a quarter of each epoch.

**Likeness was measured, not eyeballed:** ArcFace (InsightFace `buffalo_l`) cosine similarity between each preview's
face and the centroid of that person's photos, with sd-backend's tools (`projects/sd-backend/scripts/face_likeness`,
its stored `centroid.json` per ai-toolkit dataset). A match is 0.65 (sd-backend's threshold); a real photo against
the centroid of the person's other photos scores ~0.73-0.79, an unrelated face 0.0-0.15. One seed per person per
preview round, so each number carries ±0.05-0.1 of seed noise: read the trends.

**Stage 1: three people, steps 0-3200** (`repeats: 1`, ~18 passes over each photo):

| Step | 0 | 800 | 1600 | 2400 | 3200 |
|---|---|---|---|---|---|
| Anne (62 photos) | −0.03 | 0.31 | 0.51 | 0.62 | **0.68** |
| Bella (37) | 0.02 | 0.25 | 0.32 | 0.43 | 0.48 |
| Clara (35) | −0.03 | 0.21 | 0.42 | 0.58 | **0.68** |
| "a woman" (no name) vs Bella | 0.12 | 0.20 | 0.35 | 0.30 | 0.20 |

**Stage 2: five more people, resumed at 3200, to 10,000.** The resume started a fresh epoch (the dataset changed),
reloaded the three people's prior and recorded the five new ones' from the original model. Repeats from the stage-1
curves: Anne and Clara 0.3 (they had passed 0.65), Bella 1 (stuck), the new five 1.

| Step | 4000 | 5600 | 7200 | 8800 | 10000 |
|---|---|---|---|---|---|
| Anne (x0.3) | 0.69 | 0.73 | **0.75** | 0.67 | 0.59 |
| Bella (x1) | 0.51 | 0.55 | 0.63 | 0.63 | **0.64** |
| Clara (x0.3) | 0.61 | 0.58 | 0.61 | 0.63 | 0.62 |
| Dana (22) | 0.25 | 0.16 | 0.27 | 0.40 | 0.41 |
| Ella (26) | 0.24 | 0.19 | 0.24 | 0.49 | **0.59** |
| Felicia (36) | 0.03 | 0.00 | 0.01 | 0.16 | 0.20 |
| Greta (23) | 0.23 | 0.23 | 0.26 | 0.34 | 0.41 |
| Hanna (24) | 0.23 | 0.35 | 0.44 | 0.40 | 0.47 |

What the two stages showed:

- **Likeness follows passes over the photos.** The first people reached the 0.65 match at ~18 passes; the five added
  later were all still rising at 10,000 (~18 passes of theirs): stop on a flat curve, not a step count.
- **Don't cut a person who has it to 0.3.** Anne peaked at 0.75 and then lost ground (0.59) while the others trained:
  a random third of her photos per epoch was too little to hold her. Keep 0.5-1 for everyone still in the set.
- **Raising a stuck person works.** Bella sat at ~0.48 for 1200 steps at a third of the epoch's share; at the new
  people's rate she climbed to 0.64.
- **Triggers that start the same collide.** Two triggers sharing a first name (think `annabel` / `annalise`):
  the weaker one's previews matched the stronger one (0.31) better than herself (0.20). Give every person a trigger
  that shares no leading piece with another.
- **The prior keeps the class.** "photo of a woman" with no name never matched anyone above 0.35 (mid-run, then back
  to ~0.2), and at the end rendered almost the same face as the base model at the same seed.
- **Check the captions.** One set made of video frames had 5 of 24 captions calling her "a man" (a shaved head);
  fixed before the resume.
- **Similar faces drift toward each other:** at the end some newer people matched one another at 0.34-0.39 (well
  below their own scores). Distinct photos and enough passes for each are the remedy.

The final checkpoint moved each trained matrix by 1.5-3% (text fusion ~2.4% median, blocks ~1.6%); everything not
trained (input/output layers, modulation) is bit-identical to the base.

## Rendering a Krea 2 finetune

A finetune of RAW renders with **RAW settings**: Euler, scheduler `simple`, **28 steps, CFG 4.5**, empty negative,
and the training shift. In ComfyUI that shift is `ModelSamplingFlux` with **base_shift 0.5, max_shift 0.906** and the
image's real width and height (ComfyUI interpolates over 256-4096 tokens, Krea 2 over 256-6400; these two values make
them equal). With that, ComfyUI draws the same noise from the same seed and reproduces the previews almost
pixel-for-pixel (likeness 0.593 vs 0.594). Two workflows in `ComfyUI/user/default/workflows/`:
`krea2_ft_raw_preview_imitation` (this) and `krea2_ft_turbo_lora` (below).

- **RAW at Turbo settings (8 steps, CFG 1) is grey mush**, for the base and the finetune alike: not a broken model.
- **Turbo at CFG 4.5 overcooks**: Turbo is CFG-distilled.
- **Turbo with no ModelSamplingFlux is right for Turbo**: ComfyUI's Krea 2 default is a fixed mu of 1.15, the shift
  Turbo was distilled at.
- **The trigger exactly as trained**: "photo of anne, a woman ..."; a shorter or different spelling only partly
  lands on the person.
- **Changes to the text conditioning change the identity.** The finetune binds names in the text-fusion stack, so
  anything that reweights Qwen's 12 layers before or inside it moves the person too: a node scaling three layers by
  2.5-5x dropped Anne from 0.58 to 0.14, a LoRA re-mixing the 12 layers (`txtfusion.projector`) to 0.22, while a
  small LoRA on the text-fusion blocks only changed her by 0.01.

**On Turbo, fast:** the finetune's change on top of Krea 2 Turbo renders at **8-10 steps, CFG 1**, two ways:
- *an extracted LoRA:* `extract_lora.py` at rank 256 / 128 / 64 kept ~11% / 6% / 4% of the change's norm (the rest
  is noise from batch-1 steps) and still most of the likeness: Anne 0.58 on Turbo + r256 against 0.59 on the
  finetune, Clara 0.48 against 0.62.
- *at full rank:* `Turbo + 1.0 x (finetune - its base)` with `merge_lora_into_model.py --add-diff`, no SVD cut
  ([merge-lora-into-model.md](merge-lora-into-model.md)).

**Merged bases:** a 50/50 weighted sum of Kroma 0.3.1 Turbo and Krea2Realism (with LoRAs baked in) still prompts like
Turbo: **10 steps, CFG 1**. Used as a finetune base, its previews need those settings too (`sample.steps: 10`,
`sample.cfg: 1`).


## Not there yet

Compared with the MOSS-Image finetune ([moss-image-finetuning.md](moss-image-finetuning.md)):

- no EMA (an fp32 average of 12.9B weights would be 52 GB of CPU RAM)
- no on-disk encoder cache: every start encodes the dataset again
- no `[monitor]` weight drift (the trained weights move between GPU and CPU with the windows)
- the Krea 2 LoRA trainer has no previews yet (the sampler and decoder are there to add them)
- the previews always use the size-dependent shift: a Turbo-like base is previewed at its steps and CFG, not at
  Turbo's fixed mu 1.15
- one GPU only; batch size 1
