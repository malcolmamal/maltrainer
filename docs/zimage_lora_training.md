# Z-Image LoRA Training Guide

## 1. Core Concepts & "Gotchas"

Training a LoRA for **Z-Image** (Tongyi-MAI) is significantly different from standard Stable Diffusion XL or Flux training. This document covers the specific requirements uncovered during debugging.

### A. Mathematical Alignment (Critical)
Z-Image uses a specific Flow Matching formulation that differs from the `diffusers` default.

*   **Flow Target**: `latents - noise` (Standard is `noise - latents`)
*   **Timesteps**: `1.0 - t` (Inverted range)
*   **VAE Scaling**:
    *   **Scale**: `0.3611`
    *   **Shift**: `0.1159`
    *   *Note*: Standard VAEs use just a scale factor (e.g., 0.18215). Z-Image requires both shift and scale to normalize latents correctly.

### B. LoRA Structure & Prefixing
To ensure compatibility with `diffusers` and `ComfyUI` loading (which use safetensors), the LoRA keys must follow a specific naming convention.

1.  **Prefix**: `diffusion_model.`
    *   *Incorrect*: `transformer.` (files load but weights are ignored).
2.  **Keys**: `lora_A` and `lora_B`
    *   *Incorrect*: `lora_down` and `lora_up`.
3.  **Target Modules**:
    *   Includes standard attention: `to_q`, `to_k`, `to_v`, `to_out.0`.
    *   Includes feed-forward: `feed_forward.w1`, `w2`, `w3`.
    *   **CRITICAL**: Must include `adaLN_modulation.0`. This module mediates the timestep/text embedding conditioning. Without it, the model cannot "hear" the prompt or timestep.

### C. Text Encoding
The Z-Image text encoder (Qwen2-based) expects a specific "thinking" chat template.
*   **Flag**: `enable_thinking=True`
*   **Template**: Handled by the tokenizer's `apply_chat_template` method.

## 2. Configuration Reference (`config/mylora.yaml`)

```yaml
lora:
  prefix: "diffusion_model."
  rank: 32
  alpha: 32
  target_modules:
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out.0"
    - "feed_forward.w1"
    - "feed_forward.w2"
    - "feed_forward.w3"
    - "adaLN_modulation.0" # Required for conditioning

train:
  timestep_weighting: "none" # Use simple mean loss
  do_differential_guidance: false # Disable for speed (2x faster)
  optimizer: "adamw8bit"
  num_epochs: 100 # High epoch count needed for likeness
```

## 3. Troubleshooting "No Likeness"

If the loss is low (e.g., 0.4 - 0.6) but the generated images look generic:
1.  **Check Key Prefixes**: Use `inspect_lora.py` to ensure keys start with `diffusion_model.`.
2.  **Check Weights**: Use `check_weights.py` to ensure weights are not all zero.
3.  **Check Log**: Ensure `adaLN_modulation` is being trained (present in target modules).
4.  **Strength**: If likeness appears only at strength > 1.0, increase `num_epochs`.

## 4. Differential Guidance & Quality

*   **Differential Guidance**: Enabling this makes the trainer run the model **twice per step** (one conditioned, one unconditioned).
    *   **Cost**: Doubles training time (e.g., 15m -> 30m).
    *   **Benefit**: Forces the model to pay significantly more attention to the prompt/trigger word.
    *   **AI Toolkit Default**: Uses `scale: 3.0`.
    *   *Recommendation*: Always enable this for character/style LoRAs.

## 5. Base vs. Turbo Compatibility

Z-Image Turbo is a distilled version of Z-Image Base. When training LoRAs on Base and using them on Turbo, you will often encounter a "strength mismatch."

*   **Observation**:
    *   **Base**: Likeness appears at strength `1.0`.
    *   **Turbo**: Likeness requires forcing strength to `2.0` - `2.5`.
*   **Why?**: Distilled models (Turbo) have a much stiffer latent space/vector field. A small nudge (rank update) that steers Base significantly barely moves Turbo, because Turbo has been trained to minimal steps (e.g., 4-8 steps) where the trajectory is rigid.
*   **Mitigation**:
    1.  **Over-training**: Train for significantly more steps (e.g., 2000+ steps for 21 images) to bake the weights in harder.
    2.  **Higher Rank**: Increase LoRA rank (e.g., 64 or 128) to give the adapter more "muscle" to push the Turbo trajectory.
    3.  **Train on Turbo**: Ideally, train the LoRA directly on the Turbo checkpoint if that is the final target.

## 6. Config Reference (Production)

```yaml
lora:
    prefix: "diffusion_model." # REQUIRED
    rank: 32 (or 64 for Turbo compatibility)
    target_modules:
        - "adaLN_modulation.0" # REQUIRED
        - ... (standard attention/ffn)

train:
    num_epochs: 100 # Minimum for robust likeness
    do_differential_guidance: true # HIGHLY RECOMMENDED
    differential_guidance_scale: 3.0
```
