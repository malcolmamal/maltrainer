# LTX-2.5 LoRA training

`train_ltx25_lora.py` trains LoRAs for **LTX-2.5 Dev** (Lightricks, 22B audio-video DiT) from ComfyUI's
single-file checkpoints. It trains on stills, video clips and the sound of those clips, in one run. It
caches every latent and caption once, unloads the encoders, then trains on the frozen DiT.

> **Status: built and checked against ComfyUI's code, not yet run on the LTX-2.5 files.** They were not on
> disk when this was written. See [What was verified](#what-was-verified).

```bash
python train_ltx25_lora.py --config my_ltx25_config.yaml
```

```bash
run_ltx25_training.bat feliciaday              # fills templates/ltx25_template.yaml (Automagic3)
run_ltx25_training.bat feliciaday adamw8bit
```

The third way is the web UI: pick **LTX-2.5 (Dev, video + audio)**, a dataset and an optimizer.

## Model files

All from `Lightricks/LTX-2.5`, in ComfyUI's folders.

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | The **dev** model, 21.5 GB. The bf16 file loads too (about 44 GB). The `distilled` files are refused |
| `model.video_vae_path` | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | The classic conv VAE. Only its encoder runs. The default 2.5 VAE differs by a diffusion *decoder*, which training does not use |
| `model.audio_vae_path` | `vae/ltx-2.5-audio-vae-bf16.safetensors` | Only the encoder runs; the vocoder in the file is ignored |
| `model.text_encoder_path` | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (or `-bf16`) | **Gemma 4 12B** with the projection the DiT's text connectors read. The Gemma 3 12B and Gemma 4 E4B files used by LTX-2.x do **not** work |

LTX-2.5 changed the text encoder and the VAEs from LTX-2.0 and 2.3, so the older files do not fit. The
frozen DiT stays INT8 on the GPU (about 22 GB; about 17 GB without the audio stream), running through the
INT8 ConvRot kernels shared with FLUX.3 and MiniMax H3. The encoders are loaded only while caching.

## What it trains

- **Stills** train as one-frame clips. **Videos** train as clips of `dataset.clip_frames` frames (8k + 1: 9,
  17, 25, 49, 97 ...) taken from the start of the file, resampled to `dataset.fps` (24).
- A clip **with sound** also trains the audio stream: its first seconds are encoded by the audio VAE into
  exactly as many latent frames as the clip has (25 a second). Stills, silent clips and
  `dataset.audio: false` train video only.
- If **no** sample has sound, the audio half of the DiT is not even loaded.
- Captions are `.txt` files next to the media, as for the other models. A sentence works; the model was trained
  on long descriptive prompts.
- The loss is `video_loss_weight × MSE(video) + audio_loss_weight × MSE(audio)` on the velocity `noise − x0`.
  Both streams share one sigma per step.

Not supported yet: audio-only recordings, IC-LoRA / video-to-video, first-frame (image-to-video) conditioning
during training, several clips per long video, sampling previews during training.

## Template: `templates/ltx25_template.yaml`

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 32 / alpha 32. `targets: auto` = all attention (video, audio and cross-modal) when any clip has sound, otherwise the video attention only | the official LTX-2 trainer and musubi-tuner; the video-only set leaves the audio layers untouched |
| Optimizer | Automagic3 from 1e-6, no weight decay | maltrainer's FLUX.3 setup. AdamW 8-bit at 1e-4 is the official trainer's rate |
| Timesteps | `shifted_logit_normal`, "stretched" mode: a logit-normal whose mean grows linearly with the video token count (0.95 at 1024 tokens, 2.05 at 4096), its 0.5 to 99.9 percentiles mapped onto [0, 1], 10 % uniform | the official sampler of LTX-2.3 and 2.5 |
| Caption dropout | 0.05 | |
| EMA | 0.98 with warm-up; checkpoints save the average | as FLUX.3 |
| Resolution | 512 aspect buckets on a 32 px grid, 49-frame clips | |

## Configuration reference

| Key | Default | Meaning |
|---|---|---|
| `dataset.resolution` | 512 | Area target of each frame; buckets keep the aspect on a 32 px grid |
| `dataset.fps` | 24 | Frame clock of the clips (RoPE time is seconds) |
| `dataset.clip_frames` | 49 | Frames per clip, rounded down to 8k + 1 |
| `dataset.audio` | `true` | Train on soundtracks |
| `lora.targets` | `auto` | `auto`, `attn`, `video`, `attn_ff`, `video_ff` or `full` |
| `train.timestep_distribution` | `shifted_logit_normal` | or `uniform` |
| `train.shifted_logit_mode` | `stretched` | or `legacy` (`sigmoid(N(shift, std))`) |
| `train.shifted_logit_shift` | by token count | Fixes the logit-normal's mean |
| `train.shifted_logit_uniform_prob` | 0.1 | Share of uniform draws in `stretched` mode |
| `train.video_loss_weight` / `audio_loss_weight` | 1 / 1 | |
| `train.int8_grad_precision` | `bf16` | Backward through the frozen INT8 Linears |
| `train.caption_dropout` | 0 | Probability of training a step on an empty prompt |

`train.batch_size` must be 1 (clips differ in size and length); use `gradient_accumulation_steps`.

LoRA files use ComfyUI's module paths (`diffusion_model.transformer_blocks.N.attn1.to_q.lora_A.weight`, ...),
are saved in bf16 and load in ComfyUI's LTX workflows without conversion.

## What was verified

Everything below ran on CPU against ComfyUI's own modules (checkout of 2026-10-24).

| Part | Check |
|---|---|
| DiT (both streams, text and cross-modal attention, connectors, RoPE, AdaLN) | Random weights, fp32: matches ComfyUI's `LTXAVModel` to 2e-6, for split and interleaved RoPE, with and without the RoPE padding, batch 1 and 2, and for a video-only call |
| Gemma 4 text stack | `transformers`' `Gemma4TextModel` equals ComfyUI's Gemma 4 12B-style transformer to 5e-7 on every hidden state (k = v global layers, sliding window) |
| Text projection | Identical to ComfyUI's `DualLinearProjection` |
| Video VAE encoder | Real `LTX2_video_vae_bf16` weights: matches ComfyUI's to 4e-6 |
| Audio VAE encoder + mel front end | Real `LTX2_audio_vae_bf16` weights: identical at 16 kHz, 9e-5 from 44.1 kHz |
| INT8 text encoder and DiT paths | Run on tiny INT8 files of the same layout |

**Not checked**, because the files were absent: the real LTX-2.5 DiT, Gemma 4 12B and 2.5 VAEs.
- The DiT's hyper-parameters (RoPE type, timestep scales, ...) come from the file's `config` metadata, as in
  ComfyUI, falling back to LTX-2.3's values. If the 2.5 file lacks the metadata and differs from 2.3, the
  first symptom is a LoRA that trains with a normal-looking loss and does little.
- A real GPU run: speed and VRAM are unmeasured. Expect the INT8 DiT, activations and the LoRA to fit a 32 GB
  card at 512 px and 49 frames with gradient checkpointing, but that is an estimate.

The first thing to do with the files is `python train_ltx25_lora.py --config ... --dry-run`, then a
10-step run on three clips, and to compare the saved LoRA's effect in ComfyUI.
