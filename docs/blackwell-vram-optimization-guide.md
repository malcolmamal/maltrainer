# Blackwell RTX 5090 & Z-Image Training Optimization Guide (2026)

This guide documents critical optimizations and fixes discovered for training Z-Image LoRAs on high-end NVIDIA Blackwell hardware (RTX 5090) under Windows.

## 1. Environment & Hardware Setup

### RTX 5090 (Blackwell) Compatibility
For the RTX 5090, you must use **PyTorch 2.10+** and **CUDA 13.0+** to leverage the full compute power and architecture-specific optimizations.

**Installation Command:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## 2. Windows-Specific Stability

### Multiprocessing & "Pickling" Errors
Windows uses the `spawn` method for multiprocessing, which require all objects passed to worker threads (like the DataLoaders) to be serializable (picklable).
- **Issue**: Using `lambda` functions or local functions in the `ZImageDataset` will cause `AttributeError: Can't pickle local object`.
- **Fix**: Remove all VAE encoding and Tokenization logic from the `Dataset.__getitem__`. Move these operations into the main training loop.

### The "Infinite Hang" (Tokenizer Padding)
Z-Image uses the Qwen2/3 text encoder, which has a default context window of 32k tokens.
- **Issue**: If you use `padding="max_length"` without a specific limit, the GPU will attempt to process 32,000 tokens for every single image, leading to a permanent "hang" or massive VRAM usage.
- **Fix**: Always cap `max_length` in the tokenizer:
  ```python
  tokens = tokenizer(
      caption, 
      max_length=256, # Safety cap
      padding="max_length", 
      truncation=True, 
      return_tensors="pt"
  )
  ```

## 3. VRAM Optimization (RTX 5090 / 32GB)

Training a Z-Image (7B+ parameters) alongside a Text Encoder (7B+) is extremely demanding. Even on a 5090 with 32GB VRAM, the model can OOM without specific settings.

### Gradient Checkpointing
Essential for large Diffusion Transformers (DiT). It trades a small amount of compute time for significantly lower VRAM usage.
- **Implementation**:
  ```python
  transformer.enable_gradient_checkpointing()
  ```

### Precision
Always use **BFloat16** (`bf16`) training on Blackwell cards. It provides the stability of Float32 with the memory efficiency of Float16.

## 4. Advanced LoRA Tuning

### Attention vs. Feed-Forward (MLP)
- **Attention Modules** (`to_q`, `to_k`, `to_v`, `to_out.0`): Capture relationships, composition, and "where" things are.
- **MLP Modules** (`feed_forward.w1`, `.w2`, `.w3`): Capture "what" things look like (texture, color, artistic style).
- **Recommendation**: To achieve high-quality results matching professional templates (like `ai-toolkit`), target **both** sets of modules at **Rank 32/Alpha 32**. This results in a file size of approximately **170MB**.

### Epochs vs. Repeats
- **Repeats**: Multiplies the dataset *before* training. Use higher repeats (e.g., 10) for very small datasets (under 20 images) to keep the GPU utilization high.
- **Epochs**: The number of times the model sees the entire (repeated) dataset.
- **Rule of Thumb**: 100 Epochs at 1 Repeat is generally more stable than 10 Epochs at 10 Repeats for small concept training, as it increases the frequency of data shuffling.

## 5. ComfyUI Compatibility

### Key Naming Convention
ComfyUI's Z-Image loader expects LoRA keys to follow the model's internal dot-notation exactly.
- **Incorrect (Safe-name)**: `layers_0_attention_to_q.lora_down.weight` (Fails to load)
- **Correct (Dot-notation)**: `layers.0.attention.to_q.lora_down.weight` (Loads successfully)

### LoRA Alpha Metadata
Every LoRA layer in the `.safetensors` file should include an `.alpha` key. Without this, some loaders may default to a strength of 1.0/Rank, leading to extremely "weak" or "fried" looking images.
