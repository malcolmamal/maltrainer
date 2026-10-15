# Krea 2 LoRA training

`train_krea2_lora.py` trains Krea 2 LoRAs from ComfyUI's single-file checkpoints. It caches each
image's latent and each caption's embedding once, unloads the encoders, then trains on the
12.9B DiT with fp8 weights.

There are three ways to run it:

```bash
python train_krea2_lora.py --config my_krea2_config.yaml
```

```bash
run_krea2_training.bat feliciaday            # fills config/krea2_template.yaml (Automagic3)
run_krea2_training.bat feliciaday prodigy    # fills config/krea2_prodigy_template.yaml
```

The third is the web UI: pick **Krea 2**, a dataset and an optimizer.

`run_krea2_training.bat` wraps `scripts/train_from_template.py`. That script fills the template's
placeholders: the dataset name, and every model file found under `COMFYUI_MODELS_PATH`. It writes
the filled config into the output folder and starts the trainer. Pass `--max-steps`,
`--epochs`, `--trigger` or `--dry-run` after the optimizer.

## Model files

All files are ComfyUI's, the `Comfy-Org/Krea-2` repack.

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `krea2_raw_fp8_scaled.safetensors` | Train on **RAW**; Turbo is distilled. `krea2_raw_bf16.safetensors` also works: its 224 block Linears are quantized per row to fp8 on load |
| `model.vae_path` | `qwen_image_vae.safetensors` | Krea 2's VAE. `wan_2.1_vae.safetensors` has the same layout |
| `model.text_encoder_path` | `qwen3vl_4b_fp8_scaled.safetensors` | Qwen3-VL-4B; bf16 files load too |

The DiT keeps its fp8 weights and dequantizes each Linear on every call, in the forward and the
backward pass. The LoRA therefore learns against the exact weights ComfyUI runs. Memory:
- base about 13 GB;
- the text encoder (about 8 GB in bf16) and the VAE are loaded only during caching.

## How it matches ComfyUI

These were checked numerically on random weights against ComfyUI's own modules.

| Part | Convention | Check |
|---|---|---|
| DiT | `comfy/ldm/krea2/model.py`, same module names; receives sigma in [0, 1], predicts `noise - x0` | Output matches to 2.5e-7; fp8 path identical |
| Text | Qwen-Image system template, hidden states 2, 5, … 35 (unnormalized), 34-token system prefix cut from the output only | Token IDs identical to ComfyUI's `Krea2Tokenizer`; taps equal HF `hidden_states` |
| VAE | Single-frame Wan 2.1 encoder, posterior mean, `(z - mean) / std` (`Wan21` latent format) | Matches `WanVAE.encode` + `Wan21.process_in` to 2.4e-6 |
| Positions | Text tokens at (0, 0, 0), image tokens at (0, row, col); RoPE axes 32/48/48, theta 1000 | Covered by the DiT check |

Fizgig, ai-toolkit and musubi-tuner use the same conventions. They differ only in details that
don't change the maths:
- **Token order:** image first or text first.
- **Padding:** whether masked padding is used.
- **musubi:** passes sigma + 0.001 to the model, which maltrainer doesn't copy.

## Templates

### Automagic3 — `config/krea2_template.yaml` (default)

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 16 / alpha 16, all 264 Linears (`targets: all`) | Fizgig trains every Linear; it found the text-fusion adapter is "under-trained by our recipe" |
| Optimizer | Automagic3 from 1e-6, no weight decay | Fizgig / maltrainer's Qwen-Image 2.1 benchmark |
| Rate groups | attention / MLP / text fusion / I/O each adapt their own rate (`automagic_param_groups`) | Fizgig |
| Sign window | 16 steps (`automagic_polarity_history`) | Fizgig: at 8, the rate "crawled from 1e-6 to 4.3e-6 across a whole epoch" |
| Timesteps | logit-normal with the resolution-dependent `krea2_shift` (1.79 at 512 px) | musubi-tuner / Fizgig; the same curve as ComfyUI's sampler |
| EMA | 0.98 with warm-up; checkpoints save the average | Fizgig ("a higher late-epoch likeness") |
| Duration | 1500 steps, a checkpoint every 250 | Qwen-Image 2.1's Automagic3 benchmark |
| Resolution | 512 aspect buckets on a 16 px grid | Fizgig, OneTrainer |

On Krea 2, Fizgig's release notes say Automagic "matched the standard recipe rather than beating
it, and took longer". Compare the 1000 to 1500 step checkpoints before settling on a duration.

### Prodigy — `config/krea2_prodigy_template.yaml` (your OneTrainer recipe)

These are the values OneTrainer actually resolves from `krea2prodigy_template.json`. Several keys
in that file are ignored by OneTrainer's `PRODIGY` optimizer.

| Setting | Value |
|---|---|
| LoRA | rank 16, **alpha 1** (scale 1/16), `targets: blocks+txt`: the 256 attention + MLP Linears of the blocks and the text-fusion blocks (OneTrainer's `attn,ff`) |
| Optimizer | prodigyopt `Prodigy`, lr 1.0, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 decoupled, no bias correction, d0 1e-6, d_coef 1.0, slice_p 11 |
| Schedule | constant, 200-step linear warmup |
| Timesteps | unshifted logit-normal, `sigmoid(N(0, 1))` |
| Loss | plain MSE, gradient clip 1.0, no EMA |
| Duration | 100 epochs; batch 2 becomes `gradient_accumulation_steps: 2` |

What's deliberately different from OneTrainer:
- OneTrainer feeds the DiT sigma − 0.001 (an off-by-one); maltrainer feeds sigma.
- OneTrainer's base is requantized to int8 W8A8; maltrainer uses the fp8 file ComfyUI runs.
- OneTrainer's buckets snap to 64 px; maltrainer's snap to 16 px.
- Crop jitter isn't implemented.

In the web UI, choose **Epochs** = 100 for this recipe. The UI otherwise defaults to a step count.

## Configuration reference

| Key | Default | Meaning |
|---|---|---|
| `dataset.resolution` | 512 | Area target; buckets keep each image's aspect ratio on a 16 px grid |
| `dataset.aspect_mode` | `bucket` | `bucket` or `square` (centre crop) |
| `lora.targets` | `attn-mlp` | `attn-mlp` (224 block Linears); `blocks+txt` (+32 text-fusion block Linears = 256, OneTrainer and ai-toolkit); `all` (+ projector + 7 I/O = 264, Fizgig and musubi) |
| `lora.block_range` | all | Inclusive `[first, last]` of the 28 blocks |
| `train.timestep_distribution` | `krea2_shift` | `krea2_shift`, `logit_normal` (+ `timestep_logit_mean/std`, `timestep_shift`), `sigmoid` (unshifted), `uniform` (+ `timestep_shift`) |
| `train.automagic_param_groups` | `true` | One adaptive rate per module family |
| `train.automagic_polarity_history` | 16 | Automagic3 sign window |
| `train.learning_rate_warmup_steps` | 0 | Linear warmup (Prodigy's multiplier or the AdamW rate) |
| `train.ema_decay` | off | EMA of the adapters; checkpoints save the average |
| `train.caption_dropout` | 0 | Probability of training a step on the empty prompt |
| `train.max_steps` / `num_epochs` | — | `max_steps` > 0 wins |
| `output.name` | `krea2_lora` | `<name>.safetensors` for the final checkpoint, `<name>_step_N` for the others |

`train.batch_size` must be 1, because each caption keeps its own unpadded length, as in
ComfyUI. Use `gradient_accumulation_steps` for a larger effective batch.

LoRA files use ComfyUI's module paths:
- keys `diffusion_model.blocks.N.attn.wq.lora_A.weight` / `lora_B.weight` / `alpha`;
- saved in bf16;
- they load on RAW and on Turbo.
