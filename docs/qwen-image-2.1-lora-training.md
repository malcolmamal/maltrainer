# Qwen-Image 2.1 LoRA Training

maltrainer trains Qwen-Image 2.1 LoRAs **directly on the ComfyUI single-file checkpoints** — the INT8 ConvRot
transformer and text encoder plus the bf16 VAE. There is no diffusers conversion or multi-gigabyte diffusers
folder. The LoRAs it writes load in ComfyUI as they are.

**Recommended setting:** Automagic3, ~1500 steps (60–80 epochs for ~20 images), 512px, rank 16. That's ≈ 13–16
minutes on an RTX 5090, using ≈ 11 GB of VRAM. The measurements behind this are in the
[optimizer benchmark](qwen-image-2.1-optimizer-benchmark.md). The web UI picks the step count for you from the size
of your dataset ([details](#how-the-step-count-is-chosen)).

> **Resolution:** training currently runs at **square 512×512 only**. More resolutions and an aspect-ratio bucketing
> system are planned. Until then, crop your images to square yourself — see
> [Preparing a dataset](#preparing-a-dataset).

- [Requirements](#requirements)
- [Preparing a dataset](#preparing-a-dataset)
- [Training from the web UI](#training-from-the-web-ui)
- [Training from the command line](#training-from-the-command-line)
- [Config reference](#config-reference)
- [Using the LoRA in ComfyUI](#using-the-lora-in-comfyui)
- [Checking a run](#checking-a-run)
- [How it works](#how-it-works)
- [Known limitations](#known-limitations)

---

## Requirements

**GPU:** training at 512px, batch 1, rank 16 uses ≈ 11 GB. Measured speed on an RTX 5090 is 0.50–0.65 s/step
on an otherwise idle GPU. The choice of optimizer doesn't measurably change speed.

**Model files** — from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), in the usual
ComfyUI folders:

| File | ComfyUI folder |
| :-- | :-- |
| `qwen_image_2.1_int8_convrot.safetensors` (7.3 GB) | `models/diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |

The web UI finds each file anywhere under its folder, so subfolders like `diffusion_models/Qwen/` are fine.

**Python:** the regular maltrainer install (see the [README](../README.md)). The tokenizer and Qwen3-VL config are
bundled in `maltrainer/models/minimax_h3/assets/text_encoder/`, so nothing is downloaded at training time.
`automagic3` is built in; `prodigy` needs the `prodigyopt` package and `adamw8bit` needs `bitsandbytes`.

---

## Preparing a dataset

```
datasets/
└── rheaseehorn/          # folder name = default trigger word in the web UI
    ├── 001.png
    ├── 001.txt           # optional caption
    ├── 002.jpg
    └── ...
```

- **Images:** `.jpg`, `.jpeg`, `.png`, `.webp`, anywhere under the folder. ~20 good photos is enough.
- **Resolution — square 512×512 for now.** Every image is resized to 512×512. There is **no cropping and no
  aspect-ratio bucketing yet**, so a non-square image is *squashed*, not cropped. **Crop your images to square
  first.** The web UI warns you when it finds any non-square image. More resolutions and a bucketing system that
  keeps aspect ratios are planned.
- **Captions:** a `.txt` next to an image is its caption. If the caption doesn't already contain the trigger
  word, the trigger is prepended (`"rheaseehorn " + caption`). Images without a caption use
  `dataset.default_caption`, with the trigger prepended the same way.

> **Put the class word next to the trigger.** In the benchmark every caption started with `"a woman with …"`, so the
> trigger learned the face but never learned gender. `"feliciaday as a medieval knight"` then produced a *man*.
> Either caption as `"<trigger>, a woman with …"`, or always prompt with the class word:
> `"<trigger>, a woman, as a medieval knight …"`.

---

## Training from the web UI

Start the backend with two environment variables, then the frontend:

```powershell
# PowerShell
$env:DATASETS_PATH = "C:\path\to\datasets"            # folder that contains one folder per dataset
$env:COMFYUI_MODELS_PATH = "C:\path\to\ComfyUI\models" # where the three model files live
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8008

cd frontend
npm install     # first time only
npm start       # http://localhost:3003
```

In **Training Setup**:

| Field | What it does |
| :-- | :-- |
| Model | **Qwen-Image 2.1** (default) or Z-Image. Switching model resets the fields below to that model's defaults. If the Qwen model files can't be found, a red box says which ones. |
| Optimizer | Automagic3 (recommended), AdamW 8-bit, or Prodigy. Picking one fills in its learning rate. |
| Learning rate | Automagic3: `1e-6` is only its starting point, since it adapts the rate itself. AdamW: `1e-4`. Prodigy: `1.0`, which is its multiplier. |
| Train for | **Max steps** — stops exactly at N steps; the last epoch may be partial. **Leave it empty for auto** (Qwen): the UI picks a step count from your dataset size, see below. **Epochs** — N full passes over the dataset. |
| Save every | Checkpoint interval in steps (template default: 250). The final checkpoint is always saved. |
| Trigger word | Defaults to the dataset folder name. |

The **plan card** under the form comes from the backend and updates as you type. It shows total steps, epochs ×
steps per epoch, image count, an estimated time (from the measured 0.62 s/step), the output folder, and:

- **Notes** (blue): how the automatic step count was chosen.
- **Warnings** (amber): non-square images that will be squashed, a run that is too short or too long compared with
  what was benchmarked, Prodigy on Qwen, or missing model files.

**Add to queue** starts the job immediately if nothing else is running; otherwise it waits and starts automatically
when the running job finishes. **Dry run** walks through the planned steps without loading any model, which is
useful for checking the queue and progress display.

### Watching a run

Click **Open** on a job to get its live page:

[![The web UI's training job page: a Qwen-Image 2.1 run at step 143 of 1330](images/qwen21-web-ui/training-job.webp)](images/qwen21-web-ui/training-job.webp)

*A real run in progress: a 19-image dataset, Automagic3, 70 epochs (1330 steps), at step 143. Click the image to
enlarge it.*

- **Header:** the dataset, the model and optimizer, the duration, the output folder, and the path of the
  `config.yaml` (an exact copy of the settings used), plus a status badge: `queued`, `running`, `exited` or `failed`.
- **Progress bar:** the step counter (here 143 / 1330).
- **Live line:** the training progress bar with elapsed time, time remaining, speed, loss and epoch. Here it reads
  1.64 it/s, which is about 0.61 s per step, with roughly 12 minutes left at 11%. It is tagged `stderr` because the
  progress bar writes to stderr; that is normal, not an error.
- **Live output:** the full log. The startup sequence to expect is: VAE latents cached and the VAE unloaded, the text
  encoder loaded and the prompts cached, the encoder unloaded, then the transformer loaded (192 INT8 ConvRot
  layers), 192 LoRA layers injected (rank 16, 39.8 M trainable parameters), the optimizer line, and finally
  `Starting Qwen-Image-2.1 LoRA training for 70 epochs (1330 total steps)`. In this run that took about 8 seconds
  from loading the text encoder to the first step.

Two things worth checking in the log: the line `Embedding table: INT8 ConvRot, un-rotating with group size 256`
confirms the text-encoder fix is active (if it is missing, you are on an old version), and the loss, which is noisy
from step to step because each step is a single image at a random noise level. Single values anywhere from roughly
0.13 to 0.4 are normal (0.1468 at step 142 in the screenshot), so judge the trend over dozens of steps, not one number.

Closing the browser tab does not stop a run. Reopening the page replays the most recent output; the backend keeps
the last 2000 messages, so on a long run the startup lines eventually scroll out. The complete log is always in
`training.log` in the run's output folder.

### How the step count is chosen

With **Max steps left empty** the backend aims for **~70 epochs** of your dataset, rounds to the nearest 50, and
keeps the result within **1000–2500 steps** — the range the benchmark actually covered.

| Images | 70 epochs would be | Steps used | Epochs |
| --: | --: | --: | --: |
| 10 | 700 | 1000 (raised to the floor) | 100 |
| 21 | 1470 | 1450 | 70 |
| 100 | 7000 | 2500 (capped) | 25 |
| 354 | 24 780 | 2500 (capped) | 7, and the plan warns |

The target of ~70 epochs comes from the benchmark's ~21–23-image datasets, where ~1500 steps was the sweet spot.
Extending it to other dataset sizes is an extrapolation and has **not** been benchmarked, which is why the UI shows
you the reasoning and the epoch count instead of hiding it. The warnings use the benchmark too: likeness was at
roughly 70% of its plateau after 11 epochs and 90–95% after 22, so runs under ~20 epochs risk being under-trained,
and nothing above 119 epochs or 2500 steps was tested. Type a number in **Max steps**, or switch to **Epochs**,
to override the automatic choice.

The command-line template ([`templates/qwen_image_2_1_template.yaml`](../templates/qwen_image_2_1_template.yaml)) does
not do this: it has a fixed `max_steps: 1500`.

Each run gets its own folder, so nothing is overwritten:
`output/qwen_image_2_1/<dataset>-<optimizer>-<N>steps` (or `-<N>ep`), with `-2`, `-3`, … added if it already
exists. The folder holds `config.yaml` (the exact config used), `training.log`, and the checkpoints.

---

## Training from the command line

```bash
cp templates/qwen_image_2_1_template.yaml config/my_qwen_lora.yaml
# edit: replace [comfyui_models] and [person_to_train], set dataset.path
python train_qwen_image_2_1_lora.py --config config/my_qwen_lora.yaml

# walk through the planned steps without loading models
python train_qwen_image_2_1_lora.py --config config/my_qwen_lora.yaml --dry-run --dry-run-step-time 0.01
```

The template ([`templates/qwen_image_2_1_template.yaml`](../templates/qwen_image_2_1_template.yaml)) is the benchmarked
recommended setting. The web UI uses the same file.

---

## Config reference

**`model`**

| Key | Meaning |
| :-- | :-- |
| `transformer_path`, `text_encoder_path`, `vae_path` | The three ComfyUI files. |
| `tokenizer_path` | Optional; defaults to the bundled Qwen3-VL tokenizer assets. |
| `training_adapter_path` | Optional frozen LoRA attached during training (not trained, not saved). Strength: `lora.training_adapter_multiplier`. |

**`dataset`** — `path`, `resolution` (square edge; **keep it at 512**, the only size that has been tested),
`trigger`, `default_caption`, `repeats`, `num_workers`.

**`lora`** — `rank` (16), `alpha` (16), `prefix` (`diffusion_model.`), `target_modules` (default: all attention
and MLP projections in the 32 blocks — `to_q`, `to_k`, `to_v`, `to_out.0`, `gate_up`, `img_mlp.out`; 192 layers,
39.8 M parameters at rank 16), `dtype` (adapter training precision, `float32`).

**`train`**

| Key | Meaning |
| :-- | :-- |
| `max_steps` | If > 0, the run stops exactly after this many optimizer steps and overrides `num_epochs`. |
| `num_epochs` | Full passes over the dataset. Used only when `max_steps` is 0 or unset. |
| `save_every` | Checkpoint interval in steps. The final checkpoint is always written. |
| `batch_size` | Keep at **1** (see [Known limitations](#known-limitations)). |
| `gradient_accumulation_steps` | Steps count optimizer updates, so with accumulation 2, one step = 2 images. |
| `optimizer` | `automagic3`, `adamw8bit` (`adam8bit`), `prodigy`, or anything else for plain AdamW. |
| `learning_rate` | See the web UI table above for per-optimizer values. |
| `weight_decay` | Default `0.01`. |
| `warmup_steps` | Linear learning-rate warmup (0 = off). |
| `max_grad_norm` | Gradient clipping, default `1.0`. |
| `timestep_distribution` | `shift` / `sigmoid` / `logit_normal`: logit-normal σ with scale `sigmoid_scale`. Anything else: uniform σ. Qwen's resolution-dependent shift is applied either way. |
| `automagic_*` | `automagic_polarity_history` (8), `automagic_clip_threshold` (1.0), `automagic_beta2` (0.999), `automagic_eps`, `automagic_min_lr`, `automagic_max_lr`. |
| `prodigy_*` | `prodigy_d_coef`, `prodigy_d0`, `prodigy_beta2`, `prodigy_decouple`, `prodigy_use_bias_correction`, `prodigy_growth_rate`, … |
| `int8_grad_precision` | Backward pass through the frozen INT8 layers: `bf16` (default), `fp32`, or `int8`. |

**`output`** — `path`, `name` (file stem), `format` (`diffusers` = `lora_A`/`lora_B` keys, or `kohya` =
`lora_down`/`lora_up`), `dtype` (saved precision, `bfloat16`), `embed_config` (store the full config in the file's
metadata).

---

## Using the LoRA in ComfyUI

- Checkpoints: `<name>_step_<N>.safetensors` every `save_every` steps, and `<name>.safetensors` at the end.
- Keys use the ComfyUI layout, e.g. `diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight` (192 modules,
  576 tensors including the `alpha` values). They load with the standard LoRA loaders on the
  `qwen_image_2.1_int8_convrot` model.
- Training metadata (optimizer, steps, dataset size, config) is embedded in the file header.
  `scripts/inspect_lora_metadata.py` prints it.
- Qwen-Image 2.1 is guidance-distilled, so sample it without CFG. Prompt with the trigger **and** the class word.

---

## Checking a run

| Script | Use it to |
| :-- | :-- |
| `scripts/qwen_2_1_stack_probe.py --config <cfg>` | Check the whole stack before training. It scores the frozen base model's denoising loss per noise level. Healthy: ~0.2–0.4, far below the 2.2 "predict zero" baseline. Both historical bugs (below) made this score ~1.5–2.0. |
| `scripts/qwen_2_1_sample.py --config <cfg> --prompt … --lora <file>` | Generate one image, with or without a LoRA. |
| `scripts/qwen_2_1_sweep_checkpoints.py --config <cfg> --prompt …` | Sample every checkpoint of a run from one model load, plus a no-LoRA reference, as a contact sheet. The fastest way to pick a checkpoint. |
| `scripts/qwen_2_1_bench_generate.py` / `_score.py` / `_report.py` | The full benchmark: likeness (ArcFace), prompt adherence (Qwen3-VL judge), diversity, memorization, and charts. See the [benchmark doc](qwen-image-2.1-optimizer-benchmark.md#reproducing). |

**Reading the training loss:** it should *start* around 0.2–0.4, which is the base model's own level, and drift down
slowly. A loss that starts near 2 and collapses fast means the adapter is compensating for broken conditioning,
not learning the subject.

---

## How it works

1. **Pre-cache, then unload.** Every image goes through the VAE once and every caption through Qwen3-VL once.
   Both models are then freed, so only the INT8 transformer and the LoRA stay on the GPU. For ~20 images this
   takes 10–15 s.
2. **Frozen INT8 base.** The ConvRot layers run INT8 matmuls in the forward pass, reproducing ComfyUI's numerics.
   The LoRA adapters train in fp32 alongside them.
3. **Flow matching.** `x_t = (1 − σ)·x₀ + σ·ε`, target velocity `ε − x₀`, MSE loss. σ is shifted by Qwen's
   resolution-dependent exponential shift, `μ` from the image-token count (base 256 → 0.5, max 8192 → 0.9),
   `σ' = e^μ·σ / (1 + (e^μ − 1)·σ)`. At 512px that is a shift of ≈ 1.7.
4. **VAE:** RGBA, 16× spatial downsampling, 64 latent channels, per-channel mean/std normalization. RGB inputs get an
   opaque alpha channel.
5. **Text conditioning:** Qwen3-VL-8B reads
   `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
   The last decoder layer's output is taken *before* the final RMSNorm, and the 14-token system turn is dropped.
6. **Transformer:** single-stream DiT, 32 blocks, block-causal attention (text causal, the image attends to
   everything). Text tokens are modulated with t = 0 and image tokens with the sampled t. 3-axis RoPE.

The transformer, VAE and text encoder have each been checked numerically against ai-toolkit's reference
implementation. Transformer outputs match to ~1e-7, VAE encode/decode to ~1e-6 (round-trip L1 0.0066), and the text
encoder decodes coherent text through its own LM head.

### Two bugs to know about

Both loaded without any error and produced plausible numbers, and both made every LoRA train to "no likeness":

- **VAE:** the checkpoint's root-level `conv1` / `conv2` are `quant_conv` / `post_quant_conv` and were never loaded.
  Separately, each VAE stage has a parameter-free `AvgDown3D` / `DupUp3D` shortcut that was missing — invisible to
  a strict state-dict load.
- **Text encoder:** the INT8 ConvRot embedding table is stored Hadamard-rotated, like the linear layers, but the
  lookup never un-rotated it. Qwen3-VL was reading scrambled token vectors.

Both are fixed. Run `qwen_2_1_stack_probe.py` after any change to the model code.

---

## Known limitations

- **`batch_size` must be 1.** Captions are encoded one at a time to their natural length, so a batch mixing
  captions of different lengths can't be stacked. There is also no text padding mask.
- **Square 512×512 only, for now.** There is no cropping and no aspect-ratio bucketing, so non-square images are
  squashed — crop to square first. More resolutions and a bucketing system that preserves aspect ratios are planned.
  Qwen-Image 2.1's native resolution is ~1328px, so higher resolutions should improve fine detail, but time and
  memory grow with the token count and none of it has been benchmarked.
- `train.do_guidance_loss` / `do_differential_guidance` are accepted but have no effect in this trainer.
- The Z-Image trainer (also in the web UI) only runs whole epochs. A step budget for Z-Image is rounded up to the
  next full epoch, and the UI says so.
