# Training Options and OneTrainer Parity Notes

This document summarizes the main OneTrainer options that affect Z-Image Turbo training quality and performance, how they map to maltrainer, and what it would take to implement the missing pieces.

## Options and Status

### Logit-Normal Timestep Distribution
- **What it does**: Samples timesteps from a logit-normal distribution (more mass in mid timesteps, fewer in extreme tails).
- **Why it matters**: Matches Z-Image training reports and reduces instability vs uniform sampling.
- **Maltrainer status**: Implemented.
- **Config keys**:
  - `train.timestep_distribution: logit_normal`
  - `train.timestep_logit_mean` (default 0.0)
  - `train.timestep_logit_std` (default 1.0)
  - `train.timestep_shift` (default 1.0)
- **Implementation notes**: Pure math using `sigmoid(N(mean, std))` and optional shift.
- **Difficulty**: Low.
- **Expected impact**: Medium (stability, convergence quality).

### Learning Rate Warmup
- **What it does**: Linearly ramps LR from 0 to target LR over N steps.
- **Why it matters**: Helps prevent early-step spikes on sensitive models.
- **Maltrainer status**: Implemented.
- **Config keys**:
  - `train.learning_rate_warmup_steps`
- **Implementation notes**: A `LambdaLR` is sufficient for linear warmup.
- **Difficulty**: Low.
- **Expected impact**: Low to medium (stability, fewer early spikes).

### Constant Loss Weighting
- **What it does**: Multiplies total loss by a constant scalar.
- **Why it matters**: OneTrainer uses `loss_weight_strength: 5.0` with a constant loss weight function.
- **Maltrainer status**: Implemented.
- **Config keys**:
  - `train.loss_weight_fn: constant`
  - `train.loss_weight_strength: 5.0`
- **Implementation notes**: A simple scalar multiplier after loss computation.
- **Difficulty**: Low.
- **Expected impact**: Low (mostly changes effective LR scale).

## Missing Options (Not Implemented Yet)

### INT_W8A8 Transformer Quantization
- **What it does**: Quantizes weights to 8-bit with 8-bit activations (W8A8), reducing VRAM and potentially speeding up training.
- **Why it matters**: OneTrainer uses `INT_W8A8` for Z-Image Turbo transformer.
- **Maltrainer status**: Not implemented.
- **What it would entail**:
  - Select a quantization backend (torchao, quanto, or custom kernels).
  - Integrate quantized linear layers into the Z-Image transformer load path.
  - Ensure LoRA weight updates apply on top of quantized weights without accuracy collapse.
  - Add config toggles similar to OneTrainer (`transformer.weight_dtype`, `quantization.layer_filter`).
  - Validate numerics in bf16/bf32 and ensure training stability.
- **Difficulty**: High (requires kernel/backend integration and extensive validation).
- **Expected impact**: Medium to high (lower VRAM, possible speed gains).
- **Risks**: Stability regressions, training divergence, hard-to-debug kernel errors.

### Async Offloading (Enable Async Offloading)
- **What it does**: Moves layers/activations between CPU and GPU asynchronously.
- **Why it matters**: OneTrainer reduces VRAM use by offloading large components.
- **Maltrainer status**: Not implemented.
- **What it would entail**:
  - Introduce an offloading manager that hooks into forward/backward to move layers.
  - Create pinned-memory buffers and non-blocking transfers.
  - Add config controls for offload fraction and which modules to offload.
  - Ensure dataloader and optimizer step synchronization stays correct.
- **Difficulty**: High (complex runtime behavior, lots of edge cases).
- **Expected impact**: Medium (VRAM reduction, but can increase step time).
- **Risks**: Deadlocks, slowdowns, device mismatch errors.

### Activation Offloading
- **What it does**: Offloads saved activations (intermediate tensors) to CPU between forward and backward.
- **Why it matters**: Significant VRAM reduction for large transformers.
- **Maltrainer status**: Not implemented.
- **What it would entail**:
  - Wrap transformer blocks with activation checkpointing + activation offload hooks.
  - Allocate pinned buffers and manage async transfers.
  - Provide config for activation offload toggle and percent.
- **Difficulty**: High.
- **Expected impact**: High on VRAM, low to medium on speed (often slower).
- **Risks**: Complexity, performance regressions, instability in multi-GPU.

### torch.compile
- **What it does**: Compiles model graph for potentially faster execution.
- **Why it matters**: OneTrainer enables compile; may help throughput if stable.
- **Maltrainer status**: Not implemented.
- **What it would entail**:
  - Add a config flag to call `torch.compile()` on the transformer and/or LoRA wrapper.
  - Verify correctness with LoRA injected layers and custom forward arguments.
  - Provide safe fallback when compile fails.
- **Difficulty**: Medium.
- **Expected impact**: Low to medium (speed gains vary).
- **Risks**: Compile failures, runtime graph breaks, debugging complexity.

## Relative Difficulty and Impact Summary

- **Easiest to implement**: `torch.compile` (medium), then W8A8 quantization (high), then offloading (high).
- **Highest expected benefit**: Activation offloading (VRAM) and W8A8 quantization (VRAM + speed) for constrained hardware.
- **Best quality/stability impact**: Logit-normal sampling and LR warmup (already implemented).

## Recommended Next Steps (If We Resume Work)

1. Prototype `torch.compile` with a safe fallback.
2. Choose a quantization backend (torchao or quanto), add W8A8 support for transformer only.
3. Add activation offloading only after quantization is stable.
4. Keep a minimal config surface so presets stay readable.
