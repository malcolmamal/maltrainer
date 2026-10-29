# Visage-Image LoRA training

`train_visage_lora.py` trains LoRAs for **Visage-Image** ([Sherstnev/Visage-Image](https://huggingface.co/Sherstnev/Visage-Image)),
a 256 px text-to-image model for faces (237M DiT), from the single files in a ComfyUI install. It caches each image's
latent and each caption's text states once, unloads the encoders, then trains on the bf16 DiT. The whole model is
small: a run fits in a few GB of VRAM.

There are three ways to run it:

```bash
python train_visage_lora.py --config my_visage_config.yaml
```

```bash
run_visage_training.bat feliciaday              # fills templates/visage_template.yaml (Automagic3)
run_visage_training.bat feliciaday adamw          # templates/visage_adamw_template.yaml
```

The third is the web UI: pick **Visage-Image (256 px faces)**, a dataset and an optimizer (Automagic3 or AdamW; for
Prodigy, set `train.optimizer: prodigy` and `learning_rate: 1.0` yourself).

`run_visage_training.bat` wraps `scripts/train_from_template.py`. That script fills the template's
placeholders and writes the filled config into the output folder before it starts. Pass
`--max-steps`, `--epochs`, `--trigger` or `--dry-run` after the optimizer.

## Model files

| Key | File | Notes |
|---|---|---|
| `model.transformer_path` | `diffusion_models/Visage/visage-image.safetensors` | The 237M DiT, bf16, under the module names of the Hub repo's `transformer/model.safetensors` |
| `model.vae_path` | `vae/visage_vae.safetensors` | The f16 x 96-channel VAE: encoder for the cache, decoder for the previews |
| `model.text_encoder_path` | `text_encoders/qwen3_0.6b.safetensors` | Qwen3-0.6B; only its first 8 layers run |

The text encoder and the VAE are loaded only while caching (the VAE again for a preview).

## What the model is

- A single-stream DiT: 18 blocks, 768 wide, 12 heads, adaLN-Zero, SwiGLU, QK-RMSNorm, spiral RoPE on the image
  tokens. One token per latent cell, so a 256 px image is 256 tokens (a 16 x 16 grid of 96-channel latents).
- The timestep reaches the blocks through adaLN alone. The caption reaches them only through attention: 16 learned
  registers, then the caption's Qwen3 states, then learned pad vectors up to a length bucket (32, 64, 96 or 128).
- Rectified flow: `x_t = (1 - t) x0 + t noise`, the DiT predicts `noise - x0`, `t = 1` is noise.
- Captions are layer 8 of Qwen3-0.6B behind the fixed prefix "Describe the image.", rounded through float8. See the
  model's own [ARCHITECTURE.md](https://github.com/Nikita-Sherstnev/Visage-Image/blob/main/ARCHITECTURE.md) and
  [PROMPTING.md](https://github.com/Nikita-Sherstnev/Visage-Image/blob/main/PROMPTING.md).

## Captions and images

Captions are `name.txt` next to each image, as in the other trainers; `dataset.trigger` and `dataset.default_caption`
apply to images without one. The model was trained on long comma-separated captions (age, pose, hair, expression,
lighting ...) and on short casual prompts, 50/50; either style works for a LoRA, and captions longer than 128 tokens
are cut.

Images are centre-cropped to a square of `dataset.resolution` (default and training size 256). `aspect_mode: bucket`
keeps each image's aspect on a 16 px grid instead, but the model never saw non-square images.

**Batch size is 1** (every caption has its own length). The model trained at batch 128, so raise
`train.gradient_accumulation_steps`; the template uses 8.

**Caption dropout** replaces a caption's states with zeros of the same length: the unconditional branch of the
model's classifier-free guidance, exactly what it was trained with. The model used 0.2; the template uses 0.1.

## How it matches the model

Checked against the upstream package ([Visage-Image](https://github.com/Nikita-Sherstnev/Visage-Image), `visage.dit`
and `visage.vae`) with the real files from a ComfyUI install, fp32 on the CPU:

| Part | Check |
|---|---|
| DiT | Output identical (max abs difference 0.0) to upstream's `VisageDiT` for captions of 20 and 70 tokens; every tensor of the real file loads with `strict=True` |
| VAE | Encode and decode identical to upstream's `VisageVAE`; the encoder runs in fp32 (the bf16 file is exact in fp32) |
| Text | Identical to layer 8 of the full Qwen3-0.6B (`hidden_states[8]`, float8-rounded) on the real `qwen3_0.6b.safetensors`; its tokenizer ids equal ComfyUI's Qwen tokenizer's |
| Sampler | The model's recipe (shifted grid, SDE churn, CFG against a zero caption), in fp32 with CPU noise |

One deliberate difference: the sinusoidal timestep features are computed in fp32 even when the DiT is bf16, as they
are in the model's training (under autocast). The upstream package's bf16 inference computes them in bf16.

The test suite holds golden values made by the upstream modules on name-seeded tiny weights
(`tests/maltrainer/models/visage/test_golden.py`). **Not done:** a training run on the real files, so the template's
learning rates and step count are starting points.

## The LoRA file

Keys are `diffusion_model.<module>.lora_A.weight` / `lora_B.weight` / `alpha`, on the DiT's own Linears. `lora.targets`:

| Preset | Adapters |
|---|---|
| `blocks` (default) | attention (`attn.qkv`, `attn.out_proj`) and SwiGLU (`mlp.fc1`, `mlp.fc2`) of the 18 blocks: 72 |
| `attn` | attention only: 36 |
| `all` | `blocks` plus every block's adaLN modulation (`mod.1`), `text_proj`, the timestep MLP, `final_mod.1` and `out_proj`: 95 |

**Using it.** ComfyUI-Visage loads a whole model folder (`transformer/`, `vae/`, `text_encoder/`) and has no LoRA
input, so the LoRA does not load there. The module names match the DiT file's, so
`scripts/merge_lora_into_model.py visage-image.safetensors merged.safetensors lora.safetensors:1.0` bakes it into a
copy of the transformer file ([merge guide](merge-lora-into-model.md)); putting that copy into a folder
ComfyUI-Visage loads is up to you.

## Template: `templates/visage_template.yaml`

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 16 / alpha 16, `targets: blocks` | maltrainer's usual starting point |
| Optimizer | Automagic3 from 1e-6, no weight decay, per module family (attention / MLP / other), 16-step sign window | maltrainer's FLUX.3 / Fibo 2.0 setup |
| Timesteps | `visage`: `t = sigmoid(N(0, 1) + log sqrt(tokens / 64))`, i.e. mean 0.69 at 256 tokens | the model's `configs/dit.yaml` |
| Caption dropout | 0.1 | the model uses 0.2 |
| Accumulation | 8 samples per step, 1500 steps | untuned |
| EMA | 0.98 with warm-up; checkpoints save the average | as FLUX.3 |
| Checkpoints | only the final one (`save_every: 0`) | |

**`templates/visage_adamw_template.yaml`** (`adamw`) differs only in the optimizer: torch AdamW at a flat 1e-4, weight
decay 0.01, no warmup, the usual LoRA starting point; also untuned for this model.

`train.timestep_distribution` also takes `logit_normal` and `uniform`; `timestep_shift`, `min_sigma` and `max_sigma`
work as in the other trainers.

## Previews

The `sample:` section adds console keys (`w`, `m`, ... and `g` for all) that render a prompt from the live LoRA into
`<output>/samples/step<N>_<key>.png`, with the model's recipe: 50 steps, CFG 3, `sde` solver, churn 0.3, shift 3.3. Training waits
while it renders. See `maltrainer/trainers/visage/previews.py` for every setting (`steps`, `guidance`, `solver`,
`churn`, `shift`, `resolution`, `seed`, `ema`). Steps and churn are coupled: the model's author found that 80 to 120
steps at churn 0.3 wash the image out.
