# MOSS-Image LoRA training

`train_moss_image_lora.py` trains LoRAs for **MOSS-Image Base** (OpenMOSS, 5B) from ComfyUI's
single-file checkpoints. It caches each image's latent and each caption's embedding once,
unloads the encoders, then trains on the frozen DiT.

There are three ways to run it:

```bash
python train_moss_image_lora.py --config config/train_moss_image_feliciaday_automagic3.yaml
```

```bash
run_moss_image_training.bat feliciaday              # fills templates/moss_image_automagic_template.yaml (Automagic3)
run_moss_image_training.bat feliciaday prodigy      # fills templates/moss_image_prodigy_template.yaml
run_moss_image_training.bat feliciaday adamw8bit    # fills templates/moss_image_adamw_template.yaml
```

The third is the web UI: pick **MOSS-Image (Base)**, a dataset and an optimizer.

For how to lay out and caption a dataset (subfolders, `.txt` captions, several people), see
[datasets.md](datasets.md). To train the DiT's own weights instead of a LoRA, see
[moss-image-finetuning.md](moss-image-finetuning.md).

`run_moss_image_training.bat` wraps `scripts/train_from_template.py`. That script fills the
template's placeholders and writes the filled config into the output folder before it starts.
Pass `--max-steps`, `--epochs`, `--trigger` or `--dry-run` after the optimizer.

`--suggest-steps [EPOCHS]` counts the dataset's images per folder and prints the `train.max_steps`
for that many epochs (default 22), then exits without loading a model:

```bash
python train_moss_image_lora.py --config config/train_moss_image_feliciaday_automagic3.yaml --suggest-steps
```

## The `[monitor]` line

Every `train.monitor_every` steps (default 100) the log gets a summary: the mean loss by noise
band, Automagic3's direction signal per group, NaN gradient counts, s/step and peak VRAM. Three
NaN losses in a row stop the run without saving. The fields are explained in
[moss-image-finetuning.md](moss-image-finetuning.md#watching-a-run-the-monitor-line); a LoRA run
has no prior or drift field.

## Previews

The templates' `sample:` section renders two prompts (the trigger, and a trigger-free one) at
1024 px every 500 steps and before the first, with the LoRA as it is (the EMA average when EMA is
on). Nothing is saved but the PNGs in `<output.path>/samples/`. About 24 s per image. See
[moss-image-finetuning.md](moss-image-finetuning.md#previews-during-training) for the settings.

## Keys while training

When the trainer runs in a console window (the CLI or a `.bat`), two keys act at the next step:

- `s` saves a checkpoint now, as `<name>_step_<N>.safetensors`, and training continues.
- `q` saves `<name>_step_<N>.safetensors` and stops. No `final` file is written.

The web UI starts the trainer without a console, so the keys do nothing there.

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

All three templates use the same model files, LoRA (rank 16, alpha 16, `blocks`), timesteps
and 512 px buckets, so their runs compare directly.

- `templates/moss_image_automagic_template.yaml`: **Automagic3**. Launch point 1e-6. Attention, MLP and
  other layers each adapt their own rate, over an **8-step** sign window. On FLUX.3 a 16-step
  window left the rate stuck near 3e-5 and the LoRA under-trained. EMA 0.98, 20 epochs.
- `templates/moss_image_prodigy_template.yaml`: **Prodigy**, with FLUX.3's Prodigy settings.
  lr 1.0 (a multiplier on Prodigy's own estimate d), 100 warmup steps, decoupled weight decay
  0.01, beta2 0.99, bias correction and safeguarded warmup, no EMA, 20 epochs.
- `templates/moss_image_adamw_template.yaml`: **AdamW 8-bit** at a flat 1e-4 (Fizgig 8.1's rate
  for FLUX.3 and Krea 2), weight decay 0.01, EMA 0.98, as in the Automagic3 template.

The feliciaday runs built from them are `config/train_moss_image_feliciaday_automagic3.yaml`,
`_prodigy.yaml` and `_adamw.yaml`.

LoRA target presets (`lora.targets`):

| Preset | Adapters | What |
|---|---|---|
| `blocks` | 246 | attention (`proj.q/k/v/o`) + feed-forward (`proj.ffn_up_gate/ffn_down`) of the 42 joint blocks, block 22 skipped |
| `attn` | 164 | attention only |
| `all` | 263 | `blocks` + the 2 text-embedder blocks + input, timestep, modulation and output Linears |

The counts are with the default `lora.skip_blocks: [22]`; with `[]` they are 252 / 168 / 269.

### Block 22 is left out

The reference sampler runs CFG with block 22 skipped in the negative (empty-prompt) branch: the
pipeline's `uncond_skip_layers=[22]`, ComfyUI's SkipLayerGuidanceDiTSimple on layer 22. A LoRA
on any other block acts on both branches, so CFG's difference partly cancels it. A LoRA on block
22 acts on the positive branch alone, and CFG multiplies its effect by the CFG scale (3-4).
AI Toolkit and Fizgig leave the block out for that reason, and so does `lora.skip_blocks` by
default. The training forward still runs block 22; it only has no adapter. Set
`skip_blocks: []` to adapt all 42 blocks, or list other block numbers to leave out.

Saved keys are ComfyUI's module paths (`diffusion_model.layers.N.proj.q.lora_A.weight`).
ComfyUI loads them with its stock LoRA loader, including on the repacked files with a fused
`proj.qkv`.

## torch.compile

`train.compile: true` compiles each of the 42 blocks with `torch.compile` (after the LoRA wraps
their Linears; gradient checkpointing stays outside the compiled block). The templates leave it
off. `train.compile_mode` passes a mode such as `max-autotune-no-cudagraphs`; leave it unset for
the default.

Measured on the RTX 5090 (2026-10-18, feliciaday, bf16 DiT, rank 16, block 22 skipped, the
GPU shared with another process), steady-state steps after warm-up:

| Resolution | Eager | Compiled | Faster | Compile at the start |
|---|---|---|---|---|
| 512 px | 0.701 s/step | 0.681 s/step | 3% | ~48 s |
| 1024 px | 2.05 s/step | 1.91 s/step | 7% | ~25 s |

That is well below Fizgig's 26% (bf16 at 1 MP on a 5090). At 512 px a run needs ~2,400 steps
to repay the compile, so the templates leave it **off**. Turn it on for long runs or 1024 px and
up.

Inductor may log "fx graph cache unable to load compiled graph" (a torch unpickling bug when it
reads graphs cached by an earlier run). It then compiles again, and training is unaffected.

- **Compile time.** The first step compiles; the first new caption length recompiles once with
  dynamic sizes. After that, new caption lengths and buckets reuse the graph. On the GPU check
  (5 shapes) there were no further recompiles.
- **Numerics.** Compiled and eager steps agree to bf16 rounding (loss within 1e-4 relative, LoRA
  gradients within ~1%, the same as two eager kernels' drift).
- **Triton.** Inductor needs Triton. On Windows that is `pip install "triton-windows<3.7"` for
  torch 2.10 (installed in this repo's venv on 2026-10-18). Without it the run logs why and
  trains eager.

## First real run (feliciaday, Automagic3, 2026-10-17)

The run used 23 images at 512×512, 100 epochs (2300 steps), the Automagic3 template, the bf16 DiT
and the int8 text encoder.

- **Loading.** The real `moss_image_base_bf16` file stores **fused** `proj.qkv`. The LoRA still
  saves separate q/k/v adapters, which ComfyUI slices in. The text encoder logged 32 layers
  (8 full attention) and hidden 4096. Captions were 43-82 tokens.
- **Speed.** ~0.6 s/step, 23 minutes for the run.
- **Rate.** Automagic3's average rate peaked at 5.7e-5 by step 200 (MLP 8.4e-5, attention
  2.9e-5). It then decayed steadily to 1.1e-5 by the end (attention 7e-6).
- **Loss.** Mean loss went 0.578 → 0.546. It is dominated by the high-noise timesteps, so it
  says little about likeness.
- **Result.** The LoRA works but needs strength 1.4-1.5, so it is under-trained. Likeness at
  step 500 already matched step 2300, which fits the decaying rate.

Two levers are untested yet: a higher constant rate (the AdamW template) and lower-noise
timesteps (`timestep_logit_mean: 0`). Change one per run to see which one matters.

The Prodigy run was under-strength too. Stacked at 1.0 each, the Automagic3 step-500 and
Prodigy step-750 LoRAs gave the best results so far: see
[moss-image-lora-stacking-findings.md](moss-image-lora-stacking-findings.md). To keep such a
stack as one file, merge it with [scripts/merge_loras.py](merge-loras.md).
