# OneTrainer Architecture and a Multi-Model Plan for Maltrainer

> **Historical plan.** This is a design proposal from early in the project. It was not implemented as described:
> there is no `ModelAdapter`, `GenericTrainer`, `losses/` or `config/` package. What exists is described in
> [code-layout.md](code-layout.md) and [architecture.md](architecture.md).

This document explains how OneTrainer uses a single `train.py` entry point to handle many model types, and proposes a clean, modern architecture for expanding maltrainer to Flux2 (Klein 4/9), SDXL, and LTX2.

## How OneTrainer Handles Multiple Models

### Single Entry Point
- `scripts/train.py` loads a JSON config (`TrainConfig`) and calls `create.create_trainer(...)`.
- The trainer returned is model-aware and training-method-aware (LoRA, embedding, full tune).

### Factory-Based Construction
- The `modules.util.create` module builds concrete implementations based on:
  - `model_type` (Z_IMAGE, SDXL, Flux, etc.)
  - `training_method` (LORA, EMBEDDING, FULL)
- The created trainer composes multiple components:
  - `BaseModelLoader`
  - `BaseModelSetup`
  - `BaseDataLoader`
  - `BaseModelSampler`
  - `BaseModelSaver`

### Generic Trainer Loop
- `modules/trainer/GenericTrainer.py` contains the training loop.
- It delegates model-specific behavior to the component interfaces above.
- The trainer never hard-codes Z-Image or SDXL logic; it calls methods on the model and data loader.

**Conclusion**: OneTrainer is a wrapper around a factory + component system. The training loop is generic; model-specific details live in loaders/setups/data loaders/samplers.

## Proposed Maltrainer Architecture (Modern, Clean, DRY)

### Goals
- **Single training loop** that works across models.
- **Model-specific logic isolated** in model adapters and data loaders.
- **Typed, explicit configs** and minimal hidden state.

### Core Components

1. **ModelAdapter** (per model family)
   - Responsibilities:
     - Load model components (transformer/unet/text encoder/vae).
     - Provide text encoding and latent encoding methods.
     - Expose noise scheduler config and timestep sampling details.
     - Provide forward call signature normalization.

2. **Trainer (Generic)**
   - Responsibilities:
     - Data iteration, loss computation, optimizer step, logging, checkpointing.
     - Calls model adapter for encode/forward.

3. **DatasetPipeline**
   - Responsibilities:
     - Load images or video, caption handling, bucketing.
     - Return preprocessed batch dictionaries.

4. **LossModule**
   - Responsibilities:
     - Map model output and targets to loss.
     - Support model-specific targets (flow matching, v-pred, epsilon).

5. **Config Models (dataclasses)**
   - Separate config blocks for:
     - `model`, `dataset`, `train`, `lora`, `output`, `logging`.
   - Allow per-model overrides with explicit defaults.

### Folder Layout (Suggestion)

- `maltrainer/models/`
  - `base.py` (ModelAdapter interface)
  - `zimage.py`, `sdxl.py`, `flux2.py`, `ltx2.py`
- `maltrainer/datasets/`
  - `image_dataset.py`, `video_dataset.py`
- `maltrainer/trainers/`
  - `generic_trainer.py`
- `maltrainer/losses/`
  - `flowmatch.py`, `v_pred.py`, `epsilon.py`
- `maltrainer/config/`
  - `schemas/` (dataclasses)
  - `defaults/` (yaml presets)

## How to Add Flux2 Klein4/9, SDXL, LTX2

### 1) Flux2 Klein (4B/9B)
- Model adapter loads Flux2 transformer and text encoder.
- Supports flow matching and custom timestep schedules.
- Likely shares much of the Z-Image flow matching logic.

### 2) SDXL
- Adapter loads SDXL UNet + dual text encoders + VAE.
- Uses standard noise scheduler with eps or v-pred.
- Requires separate text encoder handling and attention masks.

### 3) LTX2 (Video)
- Adapter loads video transformer + audio/conditioning if used.
- Dataset pipeline must handle sequences, frame sampling, and caching.
- Loss must support temporal dimension and optional audio conditioning.

## Recommended Implementation Plan

1. **Refactor current Z-Image training into a GenericTrainer + ZImageAdapter.**
2. **Introduce a config schema layer.**
3. **Add SDXL adapter next** (most standard, best to validate architecture).
4. **Add Flux2 Klein adapters** (closer to Z-Image in flow matching).
5. **Add LTX2** (requires new video dataset + temporal loss handling).

## Coding Style Principles

- **KISS**: Keep each component focused and minimal.
- **DRY**: Shared logic in base classes, not copied across trainers.
- **Explicit over implicit**: Avoid global state and hidden behavior.
- **Typed configs**: Make model capabilities visible at compile time.
- **Small, testable modules**: Unit tests for adapters and losses.

## Practical Outcome

This design keeps a single, stable training loop while allowing new models to be plugged in with minimal duplication. It mirrors OneTrainer’s strength but is modernized with clear adapters and schema-driven configs.
