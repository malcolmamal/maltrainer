# FLUX.3 [klein] LoRA training

`train_flux3_klein_lora.py` trains LoRAs for the **base** (undistilled) FLUX.3 [klein] from
ComfyUI's single-file checkpoints. It caches each image's latent and each caption's embedding
once, unloads the encoders, then trains on the 5.6B DiT with its int8 weights.

There are three ways to run it:

```bash
python train_flux3_klein_lora.py --config my_flux3_config.yaml
```

```bash
run_flux3_klein_training.bat feliciaday              # fills templates/flux3_klein_template.yaml (Automagic3)
run_flux3_klein_training.bat feliciaday adamw8bit
```

The third is the web UI: pick **FLUX.3 [klein] (Base)**, a dataset and an optimizer.

`run_flux3_klein_training.bat` wraps `scripts/train_from_template.py`. That script fills the
template's placeholders and writes the filled config into the output folder before it starts.
Pass `--max-steps`, `--epochs`, `--trigger` or `--dry-run` after the optimizer.

## Model files

All files come from `black-forest-labs/flux-3-image-klein-comfy`.

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `diffusion_models/flux3_image_klein_int8_convrot.safetensors` | The **base** model. `flux3_image_klein_bf16` also works. The `_sd_` files are step-distilled; the trainer refuses them |
| `model.vae_path` | `vae/flux2_vae.safetensors` | The Flux2 VAE (128-channel latents, 16x down) |
| `model.text_encoder_path` | `text_encoders/qwen3vl_4b_int8_convrot.safetensors` | Qwen3-VL-4B-Instruct; `qwen3vl_4b_bf16` loads too |

The `qwen3.5_9b_flux3_prompt_upsampler` is not used. It rewrites short prompts at inference
time. BFL's card says FLUX.3 wants full scene captions (framing, subject, background, lighting)
and gives generic images from short ones. That applies to your training captions too.

Every block Linear of the DiT stays int8 on the GPU, about 6 GB. It runs through the INT8
ConvRot kernels shared with Qwen-Image 2.1 and MiniMax H3, so the LoRA learns against the
weights ComfyUI runs. The text encoder (about 5 GB) and the VAE are loaded only while caching.
Training VRAM at 512 px has not been measured yet.

## How it matches ComfyUI

These were checked numerically on CPU, against ComfyUI's own modules (checkout of 2026-10-15)
with random weights.

| Part | Convention | Check |
|---|---|---|
| DiT | `comfy/ldm/flux3/model.py`, same module names. Gets sigma in [0, 1] and predicts `noise - x0` | fp32 output matches to 6e-8, with separate and fused `qkv`. bf16 is within rounding noise (3e-3) |
| int8 DiT | ComfyUI's on-disk INT8 ConvRot format, quantized by comfy-kitchen | 1.2% from ComfyUI's int8 run of the same file. Both sit 1.1–1.5% from the fp32 original |
| Text | Instruct template without a think block. Right-padded with `<\|endoftext\|>` to a multiple of 80; pads at rope position 1. Hidden states 4, 8, …, 32, unnormalized, concatenated to 20480 | Token IDs identical to ComfyUI's `Flux3Tokenizer`. Embeddings match to 1e-6 on real and padding tokens, with transformers 4.57.1 and 5.0.0 |
| VAE | Posterior mean, 2x2 patches folded into channels, batch norm with the stored statistics | Matches ComfyUI's `AutoencoderKL.encode` to 4e-8 |
| LoRA file | `diffusion_model.<ComfyUI path>.lora_A/lora_B/alpha` | ComfyUI's own loader resolves every key, including `q/k/v_proj` on fused `qkv` files. The merged ComfyUI model reproduces maltrainer's LoRA forward to 5e-7 |

These checks do not replace a real-weight run. Training on real weights is still to be done on
the training machine.

Two details are copied from the reference on purpose:
- The DiT multiplies sigma by 1000 in the compute dtype. A bf16 run therefore embeds a
  bf16-rounded timestep, as ComfyUI and BFL's code do.
- The text window, including its padding, is part of the conditioning: the DiT attends to the
  padding tokens. Each caption keeps its own window. A dropped caption becomes the empty prompt
  in its own 80-token window, which is exactly the negative prompt CFG uses at inference. The
  diffusers trainer pads the empty prompt to the dropped caption's window instead, so that it
  can stack a batch.

## Template — `templates/flux3_klein_template.yaml`

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 16 / alpha 16, attention + MLP of every text-to-image block (`targets: blocks`, 228 adapters) | diffusers `train_dreambooth_lora_flux3_klein.py` |
| Optimizer | Automagic3 from 1e-6, no weight decay | maltrainer's Krea 2 / Qwen-Image 2.1 setup |
| Rate groups | attention / MLP / other each adapt their own rate | as Krea 2 |
| Sign window | 16 steps | as Krea 2 (batch-1 gradients) |
| Timesteps | uniform `u`, then the fixed FLUX.3 shift 6.93 | diffusers trainer; ComfyUI's `sampling_settings` and BFL's sampler use the same shift at every resolution |
| EMA | 0.98 with warm-up; checkpoints save the average | as Krea 2 |
| Caption dropout | 0.05 | the base model samples with real CFG, so the LoRA's unconditional branch should stay in step. diffusers suggests 0.1 |
| Duration | 1500 steps, a checkpoint every 250 | diffusers example |
| Resolution | 512 aspect buckets on a 16 px grid | — |

The 6.93 shift puts most training steps at high noise: the median sigma is 0.87. The inference
sampler spends 44 of its 50 steps above sigma 0.5. If likeness details stay soft, try
`timestep_distribution: logit_normal` with a lower `timestep_shift` (for example 3) and compare
checkpoints.

## Compared with Fizgig 8.1

Fizgig 8.1 (`src/fizgig/flux3/`, 15 October 2026) vendors BFL's reference DiT and autoencoder. Its
conventions match maltrainer's on every point checked: sigma in [0, 1], embedded in bf16; the
`noise - x0` target; text padded to 80-token windows with the padding attended; pads at rope
position 1; hidden states 4..32; the posterior mean with batch-norm statistics. Its LoRA keys
differ (kohya `lora_unet_…`), and ComfyUI loads both.

Where it differs:

| | Fizgig 8.1 | maltrainer |
|---|---|---|
| Base file | `flux3_image_klein_bf16`; it refuses the Comfy int8 file and requantizes itself on small cards | int8 or bf16. `train_flux3_klein_feliciaday_adamw_bf16.yaml` uses bf16 |
| Speed, 0.25 MP, RTX 5090 | 0.31 s/step on bf16 (0.26 compiled) | 0.72 s/step on int8 |
| Optimizer | adamw8bit, a flat 1e-4 or its adaptive 1e-4 to 2e-4; settings copied from Krea 2, "not yet tuned on Flux 3" | Automagic3 (the first run plateaued at ~3e-5), Prodigy, adamw8bit |
| Rank / EMA | 16 or 32 / 0.98 | 16 or 32 / 0.98 |
| Timesteps | logit-normal, then the 6.93 shift ("an assumption to verify") | uniform, then the 6.93 shift (the diffusers trainer); `logit_normal` selectable |
| Text cap | 4096 tokens, the reference's | 4096 tokens (was 512) |

Neither BFL nor diffusers publishes FLUX.3's training timestep distribution.

## Configuration reference

| Key | Default | Meaning |
|---|---|---|
| `dataset.resolution` | 512 | Area target; buckets keep each image's aspect ratio on a 16 px grid |
| `dataset.aspect_mode` | `bucket` | `bucket` or `square` (centre crop) |
| `lora.targets` | `blocks` | `blocks` (228: the 5 text, 5 image and 28 single blocks); `single` (168: the single blocks only); `all` (238: blocks plus the input, timestep, modulation and output Linears) |
| `train.timestep_distribution` | `uniform` | `uniform` or `logit_normal` (+ `timestep_logit_mean/std`) |
| `train.timestep_shift` | 6.93 | Shift applied to every draw; 1 = none |
| `train.gradient_checkpointing` | `true` | Recompute block activations in the backward pass |
| `train.int8_grad_precision` | `bf16` | Backward through the frozen int8 Linears: `bf16`, `fp32` or `int8` (not recommended) |
| `train.automagic_param_groups` | `true` | One adaptive rate per module family |
| `train.automagic_polarity_history` | 16 | Automagic3 sign window |
| `train.learning_rate_warmup_steps` | 0 | Linear warmup |
| `train.ema_decay` | off | EMA of the adapters; checkpoints save the average |
| `train.caption_dropout` | 0 | Probability of training a step on the empty prompt |
| `train.max_steps` / `num_epochs` | — | `max_steps` > 0 wins |
| `output.name` | `flux3_klein_lora` | `<name>.safetensors` for the final checkpoint, `<name>_step_N` for the others |

`train.batch_size` must be 1, because each caption keeps its own text window. Use
`gradient_accumulation_steps` for a larger effective batch.

Reference-image editing (the `image_cond` mode) is not trained. Its blocks stay frozen, and the
LoRA applies to the text-to-image path only.

LoRA files use ComfyUI's module paths:
- keys like `diffusion_model.single_blocks.N.q_proj.lora_A.weight` / `lora_B.weight` / `alpha`;
- saved in bf16;
- ComfyUI applies them to the int8 and the bf16 base file.
