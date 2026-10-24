# Fibo 2.0 LoRA training

`train_fibo2_lora.py` trains LoRAs for **Fibo 2.0 Dev** (`briaai/Fibo-2.0-dev`, 8B) from ComfyUI's
single-file checkpoints. It caches each image's latent and each caption's embedding once, unloads
the encoders, then trains on the bf16 DiT. It trains **text-to-image**. Edit training (reference
images) is not supported.

There are three ways to run it:

```bash
python train_fibo2_lora.py --config my_fibo2_config.yaml
```

```bash
run_fibo2_training.bat feliciaday              # fills templates/fibo2_template.yaml (Automagic3)
run_fibo2_training.bat feliciaday adamw8bit
```

The third is the web UI: pick **Fibo 2.0 (Dev)**, a dataset and an optimizer.

`run_fibo2_training.bat` wraps `scripts/train_from_template.py`. That script fills the template's
placeholders and writes the filled config into the output folder before it starts. Pass
`--max-steps`, `--epochs`, `--trigger` or `--dry-run` after the optimizer.

## Model files

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `diffusion_models/Fibo2/fibo2_dev_bf16.safetensors` | The **Dev** model: full, not distilled, with CFG. The INT8 ConvRot file is Edit Turbo, which is distilled and not a fine-tuning base; the trainer refuses it |
| `model.vae_path` | `vae/flux2_vae.safetensors` | The Flux2 VAE (128-channel latents, 16x down), the same file FLUX.3 and MOSS-Image use |
| `model.text_encoder_path` | `text_encoders/qwen3vl_4b_int8_convrot.safetensors` | Qwen3-VL-4B-Instruct, the same file FLUX.3 uses |

The DiT stays in bf16 on the GPU (about 16 GB). The text encoder and the VAE are loaded only while
caching.

## What the model is

- A single-stream DiT of 44 blocks, 3584 wide, over 128-channel Flux2 latents (one token per
  latent pixel). Two noise-refiner blocks run on the image alone first.
- A two-layer **Perceiver** squeezes the first of six text bundles into 256 to 512 *gist tokens*
  (more for longer prompts). The sequence is `[gist | image]`; the gist tokens are modulated at
  t = 0.
- Five blocks (6, 14, 22, 30, 38) also cross-attend to their own text bundle.
- The text is Qwen3-VL-4B's hidden states: 15 layers, each scaled to unit RMS, in six bundles of
  three layers (46080 numbers per token). The prompt is **not padded**: the gist count follows the
  true token count.

## Captions

Fibo was trained on long, structured JSON captions, and prompting with them gives the best results.
A caption file next to an image (`name.txt`) can hold either:

- a **plain sentence**, used as it is; or
- a **JSON object** in Fibo's caption format (`short_description`, `objects`, `background_setting`,
  `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `context`,
  `artistic_style`, `text_render`). The trainer cleans and compacts it as the model's pipeline
  does: empty values and the `aesthetic_score` / `preference_score` keys are dropped, and the JSON
  is written without spaces.

A caption file is used as written. `dataset.trigger` and `dataset.default_caption` only apply to
images that have no caption file, and the default caption is a plain sentence, so with JSON captions
put the trigger word inside the JSON yourself (for example in `short_description`).

An empty caption encodes as Fibo's **null caption**, the negative prompt used for CFG. This is what
`train.caption_dropout` trains on.

Not supported: the images of `text_render` entries. At inference the pipeline draws each entry's
text and shows it to Qwen3-VL; this trainer encodes the caption without those images (BRIA's
trainer draws them). Captions with `text_render` still train; the text-in-image behaviour is just
not reinforced.

## How it matches ComfyUI

Checked numerically against ComfyUI's own modules (checkout of 2026-10-24):

| Part | Convention | Check |
|---|---|---|
| DiT | `comfy/ldm/fibo2/model.py`, same module names, loads with `strict=True`. Gets sigma in [0, 1], multiplied by 1000 inside, and predicts `noise - x0` | Random weights, fp32: outputs match ComfyUI's to 2e-6 (text of 5 and 300 tokens). Every key and shape of the real `fibo2_dev_bf16.safetensors` matches. Real weights in bf16 on a 4x4 latent: 6.7 % relative difference from ComfyUI's, while each sits 5.8 % (maltrainer) and 6.5 % (ComfyUI) from an fp32 run of the same weights, so the difference is bf16 rounding |
| Text | `comfy/text_encoders/fibo2.py`: instruct template, hidden layers 5 to 35 in six bundles, unit RMS | Same shapes as ComfyUI's on the real INT8 file; cosine similarity of the full embedding 0.99994 to 0.99999 (INT8 and bf16 kernel differences on CPU) |
| LoRA file | `diffusion_model.<ComfyUI path>.lora_A/lora_B/alpha` | Adapters sit on the file's fused Linears (`attention.qkv`, `feed_forward.w13`, `cross_attention.to_kv`). ComfyUI's own `load_lora` resolved all 199 adapters of a LoRA trained on the real weights, and every shape fits the real checkpoint |

## Template: `templates/fibo2_template.yaml`

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 16 / alpha 16, attention + SwiGLU + the injection blocks' cross-attention of the 2 refiner and 44 main blocks (`targets: blocks`, 199 adapters) | BRIA's script adapts every Linear at rank 64; this is the same families at a lighter rank |
| Optimizer | Automagic3 from 1e-6, no weight decay | maltrainer's FLUX.3 / Krea 2 setup |
| Timesteps | `bria`: sigma = sigmoid(N(shift, 1)), shift rising linearly from 0.5 at 256 tokens to 1.15 at 4096; 10 % uniform | BRIA's `fine_tuning/train_lora.py` (the Flux2 scheduler's `mu` range) |
| Caption dropout | 0.1 | BRIA |
| EMA | 0.98 with warm-up; checkpoints save the average | as FLUX.3 |
| Duration | 1500 steps, a checkpoint every 250 | as FLUX.3 |
| Resolution | 1024 aspect buckets on a 16 px grid | the model's training area. 768 or 512 are faster and lighter |

## Configuration reference

| Key | Default | Meaning |
|---|---|---|
| `dataset.resolution` | 1024 | Area target; buckets keep each image's aspect ratio on a 16 px grid |
| `dataset.aspect_mode` | `bucket` | `bucket` or `square` (centre crop) |
| `lora.targets` | `blocks` | `blocks` (the 2 refiner and 44 main blocks: attention, SwiGLU, injection cross-attention); `main` (the 44 main blocks only); `all` (blocks plus the Perceiver, every adaLN modulation and the input, timestep and output Linears) |
| `train.timestep_distribution` | `bria` | `bria`, `uniform` or `logit_normal` (+ `timestep_logit_mean/std`) |
| `train.timestep_uniform_mix` | 0.1 | `bria` only: the share of uniform draws |
| `train.timestep_shift` | 1.0 | Flow shift applied to every draw; 1 = none |
| `train.gradient_checkpointing` | `true` | Recompute block activations in the backward pass |
| `train.automagic_param_groups` | `true` | One adaptive rate per module family (attention / MLP / other) |
| `train.automagic_polarity_history` | 16 | Automagic3 sign window |
| `train.ema_decay` | off | EMA of the adapters; checkpoints save the average |
| `train.caption_dropout` | 0 | Probability of training a step on the null caption |
| `train.max_steps` / `num_epochs` | — | `max_steps` > 0 wins |
| `output.name` | `fibo2_lora` | `<name>.safetensors` for the final checkpoint, `<name>_step_N` for the others |

`train.batch_size` must be 1, because each caption has its own gist-token count. Use
`gradient_accumulation_steps` for a larger effective batch.

LoRA files use ComfyUI's module paths (`diffusion_model.layers.N.attention.qkv.lora_A.weight`, …),
are saved in bf16 and load in ComfyUI's Fibo 2.0 workflows without conversion.

## Not yet done

- A run on a GPU. The trainer was run end to end on CPU with the real weights (3 steps at 128 px,
  about 14 s per step, finite loss, a loadable LoRA), but speed and VRAM at 1024 px are not
  measured. BRIA reports 31 GiB at 1024 px, rank 128 and 8 gradient-accumulation steps on an L40S.
- Edit training, and the `text_render` images.
- INT8 ConvRot DiT weights: ComfyUI ships those only for Edit Turbo.
