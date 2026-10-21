# Architecture Findings and Roadmap (Maltrainer)

> **Historical plan.** This is a design proposal from early in the project. It was not implemented as described:
> there is no `ModelAdapter`, `GenericTrainer`, `losses/` or `config/` package. What exists is described in
> [code-layout.md](code-layout.md) and [architecture.md](architecture.md).

This is a working document to capture findings and recommended next steps so we can resume later with full context.

## What We Implemented So Far

- Added support for Z-Image Turbo training variants:
  - De-distilled transformer path via `model.name_or_path` with `model.extras_name_or_path` for TE/VAE.
  - Training adapter merge via `model.assistant_lora_path`.
- Added Z-Image Turbo templates:
  - `templates/zimage_turbo_template-onetrainer.yaml` (de-distilled)
  - `templates/zimage_turbo_template-adapter.yaml` (training adapter)
- Added Prodigy optimizer support for LoRA + embedding.
- Added logit-normal timestep sampling and LR warmup.
- Added constant loss weighting support.
- Added docs:
  - `docs/training-options-and-parity.md`
  - `docs/onetrainer-architecture-and-multi-model-plan.md`

## Verified OneTrainer Turbo Config (Feliciaday)

From `zimage_feliciaday_onetrainer_adamw_turbo.json`:
- `lora_rank: 16`
- `lora_alpha: 1.0`
- `learning_rate: 3e-4`
- `batch_size: 2`
- `timestep_distribution: LOGIT_NORMAL`
- `learning_rate_warmup_steps: 200`
- `loss_weight_fn: CONSTANT`, `loss_weight_strength: 5.0`
- `transformer.model_name`: `.../z_image_de_turbo_v1_bf16.safetensors`
- `transformer.weight_dtype: INT_W8A8`
- `text_encoder.weight_dtype: FLOAT_8`
- `compile: true`
- `enable_async_offloading: true`
- `enable_activation_offloading: true`

## What is Implemented in Maltrainer

- `train.timestep_distribution: logit_normal`
- `train.learning_rate_warmup_steps`
- `train.loss_weight_fn: constant` + `train.loss_weight_strength`
- `model.extras_name_or_path` for TE/VAE
- `model.assistant_lora_path` to merge training adapter

## What is Not Implemented Yet

- Transformer quantization to W8A8 (INT_W8A8)
- Text encoder FLOAT_8 quantization
- Async offloading
- Activation offloading
- torch.compile

## OneTrainer Architecture (Short Summary)

- A single `scripts/train.py` entry point loads `TrainConfig` and calls `create.create_trainer(...)`.
- `create_trainer` builds a model-specific trainer using factory methods.
- `GenericTrainer` is the core training loop that delegates model-specific behaviors to:
  - `BaseModelLoader`
  - `BaseModelSetup`
  - `BaseDataLoader`
  - `BaseModelSampler`
  - `BaseModelSaver`
- Conclusion: OneTrainer uses a factory + component architecture; the loop is generic.

## AI Toolkit Architecture Notes

- Model-specific logic lives in `extensions_built_in/diffusion_models/*`.
- Config is heavily model-driven via `toolkit/config_modules.py`.
- Turbo variants use:
  - `model.extras_name_or_path` to load TE/VAE separately.
  - `model.assistant_lora_path` for training adapter.

## Best Borrow Ideas (Cross-Project)

From AI Toolkit:
- Model config surface: `extras_name_or_path`, `assistant_lora_path`, `qtype`, etc.
- Per-model modules and UI defaults for stable presets.

From OneTrainer:
- Generic trainer + component factories.
- Strong config schema and validation.

From kohya/sd-scripts:
- Dataset bucketing utilities and LoRA key conventions.

## Suggested Architecture for Maltrainer (Target State)

A modern, clean, DRY/KISS setup:

- `maltrainer/models/`
  - `base.py` (ModelAdapter interface)
  - `zimage.py`, `sdxl.py`, `flux2.py`, `ltx2.py`
- `maltrainer/trainers/`
  - `generic_trainer.py`
- `maltrainer/datasets/`
  - `image_dataset.py`, `video_dataset.py`
- `maltrainer/losses/`
  - `flowmatch.py`, `v_pred.py`, `epsilon.py`
- `maltrainer/config/`
  - `schemas/` (dataclasses)
  - `defaults/` (yaml presets)

## Suggested Implementation Order

1. Refactor current Z-Image training into a `GenericTrainer` + `ZImageAdapter`.
2. Add SDXL adapter (most standard; validates architecture).
3. Add Flux2 Klein 4/9 adapter (similar to Z-Image flowmatch patterns).
4. Add LTX2 (requires video dataset pipeline and temporal loss handling).

## Pending High-Impact Features

1. **W8A8 transformer quantization**
   - Needs a quantization backend (torchao/quanto/custom).
   - Must be compatible with LoRA updates.

2. **Async / activation offloading**
   - Requires explicit device transfer management and pinned memory.
   - Large impact on VRAM, but increases complexity.

3. **torch.compile**
   - Lower complexity; add as optional feature with safe fallback.

## Notes for Future Work

- Keep config surface explicit.
- Avoid hidden behaviors; prefer small, testable components.
- Build adapters for new models rather than new scripts.
- Ensure LoRA export stays compatible with ComfyUI and AI Toolkit naming.
