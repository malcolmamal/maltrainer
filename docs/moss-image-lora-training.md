# MOSS-Image LoRA training

`train_moss_image_lora.py` trains LoRAs for **MOSS-Image Base** (OpenMOSS, 5B) from ComfyUI's
single-file checkpoints. It caches each image's latent and each caption's embedding once,
unloads the encoders, then trains on the frozen DiT.

There are three ways to run it:

```bash
python train_moss_image_lora.py --config config/train_moss_image_feliciaday_automagic3.yaml
```

```bash
run_moss_image_training.bat feliciaday              # fills config/moss_image_template.yaml (Automagic3)
run_moss_image_training.bat feliciaday prodigy      # fills config/moss_image_prodigy_template.yaml
```

The third is the web UI: pick **MOSS-Image (Base)**, a dataset and an optimizer.

`run_moss_image_training.bat` wraps `scripts/train_from_template.py`. That script fills the
template's placeholders and writes the filled config into the output folder before it starts.
Pass `--max-steps`, `--epochs`, `--trigger` or `--dry-run` after the optimizer.

## Model files

All files come from `Comfy-Org/MOSS-Image`. ComfyUI loads MOSS-Image natively from the
2026-10 update onwards (`comfy/ldm/moss_image`).

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `diffusion_models/moss_image_base_bf16.safetensors` | 10 GB. `moss_image_base_int8_convrot` also works. The `_sft_` files are OpenMOSS's fine-tune of the base; the trainer warns |
| `model.vae_path` | `vae/flux2_vae.safetensors` | The Flux2 VAE, the same file FLUX.3 uses (`flux2-vae.safetensors` in the repo) |
| `model.text_encoder_path` | `text_encoders/qwen3.5_9b_int8_convrot.safetensors` | Qwen3.5-9B. A plain bf16 file also works. The `w4a8_gptq` file is not supported |

The text encoder runs only while prompts are cached. Its Linears stay int8 on the GPU (~10 GB),
and it is unloaded before the DiT loads.

## How it matches ComfyUI

Checked numerically against ComfyUI on CPU, with small random-weight models in ComfyUI's file
layouts:

| Part | Reference | Relative difference |
|---|---|---|
| DiT, fp32 (plain and fused-qkv files) | `comfy.sd.load_diffusion_model` → `apply_model` | ~7e-7 |
| DiT, bf16 | ComfyUI's own bf16 drift from fp32 is 6.6e-3; maltrainer's is the same | — |
| Saved LoRA, applied by `comfy.lora` (plain and fused-qkv files) | the training forward | ~1e-6 |
| Text encoder, fp32 / bf16 / int8 | ComfyUI's `MOSSImageTEModel` | 2e-7 / 2e-7 / 5e-7 |
| Text encoder, fp32 | transformers' `Qwen3_5TextModel` | 3.5e-7 |
| Token ids (template, empty prompt, newlines, CJK, emoji, special-token text) | ComfyUI's `MOSSImageTokenizer` | identical |

Conventions it follows:

- **DiT input.** The Flux2 VAE's 128-channel latent, one token per 16 px, joined with the text
  as `[image | text]`. All 42 blocks take one shared modulation of the timestep, text tokens
  included.
- **Timestep.** The DiT gets sigma in [0, 1]. Its sinusoidal embedding is of 1000 × sigma, with
  phases in float64, as in the training code.
- **Target.** The flow velocity `v = noise − x0`, under a plain per-token MSE (report equation 3).
- **Text.** The prompt is stripped and wrapped in Qwen's chat template with thinking enabled:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`. The encoder output
  is every token's final hidden state after the last norm, at the prompt's true length (no
  padding, capped at 4096 tokens). The DiT's RoPE follows that length, so training runs at batch
  size 1. Use `gradient_accumulation_steps` for larger batches.
- **Encoder precision.** As in ComfyUI, the encoder's activations run in fp32 while its weights
  stay int8/bf16. An all-bf16 run drifts ~3× further from fp32.
- **Caption dropout** encodes the empty prompt. That is the negative prompt the reference
  sampler's CFG uses (scale 3, 50 steps, shift 5).

## Timesteps

The defaults reproduce the technical report's last pretraining interval (appendix A.2, Table 6):

- logit(sigma) ~ N(1.09, 0.95)
- truncated at sigma ≤ 0.95 and rescaled onto [0, 1]
- rounded up onto the 1000-step grid
- mixed with 1 % uniform
- no shift

The sampler's mean sigma is 0.745 against 0.744 for the report's equation 4. P(sigma > 0.9) is
0.217 against 0.215.

This distribution favours high noise: only ~2 % of steps fall below sigma 0.3. If a likeness
LoRA learns composition but not fine detail, try `timestep_logit_mean: 0` (centred on sigma 0.5)
and compare checkpoints.

| Key | Default | Meaning |
|---|---|---|
| `train.timestep_distribution` | `logit_normal` | or `uniform` |
| `train.timestep_logit_mean` / `_std` | 1.09 / 0.95 | logit-normal on sigma; positive = noisier |
| `train.timestep_cutoff` | 0.95 | truncate at sigma ≤ cutoff, then divide by it; 1 = off |
| `train.timestep_uniform_mix` | 0.01 | share of uniform draws |
| `train.timestep_shift` | 1.0 | `s·σ / (1 + (s − 1)·σ)`; 1 = none |
| `train.timestep_grid` | 1000 | round sigma up onto {1/N … 1}; 0 = continuous |

## Templates

Both templates use the same model files, LoRA (rank 16, alpha 16, `blocks`), timesteps and
512 px buckets, so their runs compare directly.

- `config/moss_image_template.yaml`: **Automagic3**. Launch point 1e-6. Attention, MLP and
  other layers each adapt their own rate, over an **8-step** sign window. On FLUX.3 a 16-step
  window left the rate stuck near 3e-5 and the LoRA under-trained. EMA 0.98, 1500 steps.
- `config/moss_image_prodigy_template.yaml`: **Prodigy**, with FLUX.3's Prodigy settings.
  lr 1.0 (a multiplier on Prodigy's own estimate d), 100 warmup steps, decoupled weight decay
  0.01, beta2 0.99, bias correction and safeguarded warmup, no EMA, 100 epochs.

`config/train_moss_image_feliciaday_automagic3.yaml` (2500 steps) and
`config/train_moss_image_feliciaday_prodigy.yaml` (100 epochs) are the feliciaday runs built
from them.

LoRA target presets (`lora.targets`):

| Preset | Adapters | What |
|---|---|---|
| `blocks` | 252 | attention (`proj.q/k/v/o`) + feed-forward (`proj.ffn_up_gate/ffn_down`) of the 42 joint blocks |
| `attn` | 168 | attention only |
| `all` | 269 | `blocks` + the 2 text-embedder blocks + input, timestep, modulation and output Linears |

Saved keys are ComfyUI's module paths (`diffusion_model.layers.N.proj.q.lora_A.weight`).
ComfyUI loads them with its stock LoRA loader, including on the repacked files with a fused
`proj.qkv`.

## Not verified yet

No real-weight run has been done here (this machine has no GPU and only stub model files). For
the first run, watch:

- **VRAM.** The bf16 DiT is ~10 GB, plus activations with gradient checkpointing.
- **Step time.**
- **The loss level and its trend.**
- **Text-encoder shape check.** The real `qwen3.5_9b_int8_convrot` file must show 32 layers
  (8 full attention) and hidden 4096 in the log line `MOSS-Image text encoder: Qwen3.5, ...`.
