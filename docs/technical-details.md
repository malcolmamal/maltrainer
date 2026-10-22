# Technical Details and Usage

This document provides additional technical information about the Z-Image trainer implementation, including hardware requirements, model architecture details, and training progress visualization.

## Hardware Requirements

### GPU and CUDA Support

The Z-Image trainer is built on PyTorch and supports GPU acceleration via CUDA for significantly faster training. Here's how it works:

- **GPU Detection**: The trainer automatically detects available CUDA-compatible GPUs and uses them if present. If no GPU is found, it falls back to CPU training (though this is much slower and not recommended for practical use).

- **CUDA Requirements**:
  - CUDA 11.8 or later (recommended for compatibility with PyTorch 2.x)
  - cuDNN 8.6 or later
  - Compatible NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for larger models)

- **Multi-GPU Support**: The trainer uses PyTorch's `accelerate` library for distributed training across multiple GPUs. Set the `num_processes` parameter in your config to the number of GPUs you want to use.

- **Memory Usage**: 
  - LoRA training: Typically 8-16GB VRAM depending on batch size and model size
  - Textual Inversion: Lower memory usage (4-8GB VRAM) since only embeddings are trained

- **Checking GPU Availability**: You can verify GPU usage by monitoring your GPU utilization with tools like NVIDIA-SMI or in your training logs, which will show "cuda" if GPU is being used.

**Note**: Training on CPU is possible but extremely slow. For Z-Image models, expect training times measured in days rather than hours on CPU.

## Model Architecture

### Attention Mechanism

In the context of Z-Image training, "attention" refers to the transformer attention mechanism that forms the core of the diffusion model:

- **Transformer Attention**: Z-Image uses a transformer-based architecture similar to other modern diffusion models. The attention mechanism allows the model to focus on different parts of the input sequence (text tokens or latent representations) when generating or processing information.

- **Multi-Head Self-Attention**: The model employs multi-head self-attention layers where:
  - **Queries, Keys, Values**: Each input token is transformed into query (Q), key (K), and value (V) vectors
  - **Attention Scores**: The model computes how much each token should attend to others using scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T / √d_k) * V`
  - **Multi-Head**: Multiple attention heads operate in parallel and are concatenated, allowing the model to capture different types of relationships

- **Cross-Attention**: For text-to-image generation, the model uses cross-attention between text embeddings and image latents, allowing the diffusion process to be conditioned on text prompts.

- **LoRA Targeting**: In LoRA training, we inject trainable low-rank matrices into the attention layers (specifically the query, key, value, and output projections) to adapt the model's attention behavior for specific concepts or styles.

- **Textual Inversion**: The embedding vectors are injected into the text encoder's attention layers, allowing new concepts to be represented and attended to during generation.

**Why Attention Matters**: The attention mechanism enables the model to understand complex relationships between text and image elements, making it crucial for high-quality text-to-image generation.

## Training Progress and Visualization

### Progress Tracking

The trainer provides several ways to monitor training progress:

- **Console Logging**: Both training scripts use Python's logging module with configurable verbosity levels. You'll see:
  - Epoch and step progress
  - Loss values for each step
  - Learning rate changes
  - Time estimates for completion

- **Progress Bars**: Using the `tqdm` library, you'll see progress bars for:
  - Dataset loading
  - Training epochs
  - Validation steps (if enabled)

- **Metrics Logging**: The trainer logs key metrics including:
  - Training loss (flow matching loss for Z-Image)
  - Learning rate
  - Gradient norms (if gradient clipping is enabled)
  - Memory usage

### Visualization Options

Currently, the trainer does not include built-in visualization tools, but here are the available options:

- **Basic Progress**: Real-time console output shows training progress, loss curves, and estimated time remaining.

- **Log Files**: All output is saved to log files (configurable via `--log_file` parameter) for later analysis.

- **Integration with External Tools**:
  - **TensorBoard**: You can add TensorBoard logging by installing `tensorboard` and modifying the training scripts to use `torch.utils.tensorboard.SummaryWriter`.
  - **Weights & Biases (W&B)**: Integrate with W&B by installing `wandb` and adding logging calls to track experiments, hyperparameters, and metrics.
  - **MLflow**: For experiment tracking and model versioning.

- **Sample Generation**: During training, you can configure periodic sample generation to visualize how the model is learning. Set the `sample_interval` parameter to generate images at regular intervals.

- **Custom Visualization**: The modular design allows you to add custom visualization code, such as plotting loss curves or generating comparison images.

### Example: Adding TensorBoard Support

Maltrainer does not log to TensorBoard (or W&B, or MLflow) out of the box: its monitor writes `training.log`, the console
progress, and the preview/gallery images. The snippet below shows how you could add TensorBoard yourself.

To add basic TensorBoard logging, you can modify the training scripts:

```python
from torch.utils.tensorboard import SummaryWriter

# In the trainer class __init__:
self.writer = SummaryWriter(log_dir=config.output_dir / 'tensorboard')

# In the training loop:
self.writer.add_scalar('Loss/train', loss.item(), global_step)
self.writer.add_scalar('LR', lr_scheduler.get_last_lr()[0], global_step)
```

Then run `tensorboard --logdir=path/to/output/tensorboard` to view the training curves.

### Monitoring Best Practices

- **GPU Utilization**: Use `nvidia-smi` or similar tools to monitor GPU memory and utilization
- **System Resources**: Monitor CPU, RAM, and disk I/O during training
- **Early Stopping**: Watch for loss convergence to avoid overfitting
- **Checkpoint Validation**: Regularly validate checkpoints by generating sample images

The trainer is designed to be lightweight and focused on training, with visualization left to external tools for flexibility.

## Gradient Accumulation

### ELI5 Explanation

Imagine you're trying to learn how to throw a ball accurately at a target. Normally, you might throw the ball once, see where it lands, and then adjust your throw for the next one. But what if you could watch yourself throw the ball 4 times in a row, remember all the mistakes you made in those throws (like throwing too high, too low, left, or right), and then make one big adjustment to fix all the problems at once? That's basically what gradient accumulation does!

In machine learning training:
- **Normal training**: After each small group of examples (a "mini-batch"), the model looks at how wrong it was and makes a small improvement.
- **With gradient accumulation**: The model waits for several mini-batches (say, 4 of them), collects all the "mistakes" (gradients) from each one, adds them up, and then makes one bigger improvement.

This is super useful when you don't have enough computer memory to process a really big group of examples all at once. Instead of using a huge batch that might crash your computer, you use smaller batches but accumulate their learning over a few steps before updating the model.

### Technical Details

Gradient accumulation is a memory-saving technique that allows you to effectively use a larger batch size without increasing GPU memory usage:

- **How it works**: Instead of updating model weights after each mini-batch, gradients are accumulated (summed) over multiple mini-batches before performing the weight update.
- **Effective batch size**: If you accumulate gradients over N steps, your effective batch size becomes N × mini_batch_size.
- **Memory efficiency**: Only the gradients are accumulated, not the activations or intermediate computations, keeping memory usage low.
- **When to use**: Useful when you want the stability of large batch training but are limited by GPU memory constraints.

In the Z-Image trainer, the `gradient_accumulation_steps` parameter controls this. For example, if set to 4, the model will process 4 mini-batches, accumulate their gradients, and then update the weights once, effectively training as if it had a 4x larger batch size.

## Epochs vs Steps

### Understanding Epochs and Steps

Machine learning training can be measured in two main ways: **epochs** and **steps**. Understanding the difference is crucial for controlling your training duration.

- **Epoch**: One complete pass through your entire dataset. If you have 100 training images, 1 epoch means the model has seen all 100 images once.
- **Step**: One gradient update (weight adjustment) of the model. A step typically processes one batch of data. If your batch size is 10, then 1 step = 10 images processed.

**Key Relationship**: Number of steps per epoch = Total dataset size ÷ Batch size

### Why Some Tools Use Epochs, Others Use Steps

- **Epochs-based training**: Easier to reason about in terms of dataset coverage. "Train for 10 epochs" means the model sees the full dataset 10 times. This is good when you want consistent coverage of your data regardless of dataset size.
- **Steps-based training**: More precise control over total training time. "Train for 1000 steps" means exactly 1000 weight updates, regardless of dataset size. This is useful for comparing experiments or when you have very large datasets where epochs would take too long.

The Z-Image trainer uses `num_epochs` because it provides more intuitive control for typical fine-tuning scenarios where you want to ensure the model has seen the dataset a certain number of times.

### How Dataset Size Affects Training

The number of images in your dataset directly affects how many steps are in each epoch, which impacts total training time and behavior:

#### Example: 25 Images vs 100 Images

Assume batch_size = 5 for both cases:

**With 25 images:**
- Steps per epoch = 25 ÷ 5 = 5 steps
- If num_epochs = 10: Total steps = 5 × 10 = 50 steps
- Training time: Faster (fewer total steps)
- Learning behavior: Model sees each image more frequently per epoch (higher "repetition rate")

**With 100 images:**
- Steps per epoch = 100 ÷ 5 = 20 steps  
- If num_epochs = 10: Total steps = 20 × 10 = 200 steps
- Training time: Slower (more total steps)
- Learning behavior: More diverse examples per epoch, potentially better generalization

#### Key Implications

- **More images**: Longer training time per epoch, but potentially better learning due to more diverse data exposure
- **Fewer images**: Faster training, but risk of overfitting since the model sees the same images repeatedly
- **Same num_epochs**: Larger datasets = more total training steps = more training time
- **Balancing act**: For small datasets (like 25 images), you might need more epochs or data augmentation. For large datasets (like 100+ images), fewer epochs may suffice.

**Recommendation**: Start with num_epochs around 10-50 and adjust based on your dataset size and convergence. Monitor validation loss to determine if you need more epochs.

## Inference Script Code Review

### Textual Inversion Implementation for Z-Image

**Status**: Experimental - May be the first implementation of textual inversion for Z-Image model.

#### Core Question: Can Textual Inversion Work for Z-Image?

**Answer**: Yes, theoretically possible. Z-Image uses Qwen3 as its text encoder, which supports:
- Token addition to vocabulary
- Embedding layer resizing
- Input embedding weight manipulation

These are the same operations used in textual inversion for other transformer-based models.

#### Issues Found in Initial Implementation

1. **Pipeline Component Access**:
   - Original code assumed `pipeline.text_encoder` and `pipeline.tokenizer` exist
   - Z-Image's FluxPipeline was designed for T5, not Qwen components
   - Pipeline may not expose Qwen tokenizer/text_encoder in standard attributes

2. **Component Loading Requirements**:
   - Z-Image requires separate loading of Qwen2Tokenizer and Qwen2ForCausalLM
   - Must load from model subfolders (`tokenizer/`, `text_encoder/`)
   - Cannot rely on pipeline's built-in components

3. **Integration Challenges**:
   - FluxPipeline may not allow simple attribute replacement (`pipeline.tokenizer = ...`)
   - Internal pipeline references might not update properly
   - Generation process might not use replaced components

#### Fixes Applied

- **Imports**: Changed to `Qwen2Tokenizer` and `Qwen2ForCausalLM` (Qwen3 is based on Qwen2 architecture)
- **Component Loading**: Modified `load_textual_inversion_embedding()` to load Qwen components separately
- **Pipeline Modification**: Added code to replace pipeline's tokenizer/text_encoder attributes

#### Remaining Technical Concerns

1. **Pipeline Compatibility**:
   - Uncertain if attribute replacement (`pipeline.tokenizer = tokenizer`) works
   - FluxPipeline might have hardcoded internal references
   - May require custom pipeline subclass or manual inference

2. **Embedding Dimensions**:
   - Must verify Qwen3's embedding dimension matches saved vectors (2560 expected)
   - Multiple embedding vectors need proper token handling
   - Token positioning in vocabulary needs validation

3. **Inference Integration**:
   - Pipeline's `__call__` method may not use modified components
   - May need to override text encoding in pipeline
   - Potential conflicts with LoRA loading order

4. **Testing Requirements**:
   - No existing implementations to reference
   - Need empirical testing of token addition
   - Must verify embedding injection affects generation
   - Test combinations with LoRA

#### Alternative Approaches if Current Method Fails

1. **Custom Inference Function**:
   ```python
   def encode_text_custom(pipeline, prompt):
       # Manually encode with modified text_encoder
       inputs = pipeline.tokenizer(prompt, return_tensors="pt")
       with torch.no_grad():
           text_embeds = pipeline.text_encoder(**inputs).last_hidden_state
       return text_embeds
   ```

2. **Pipeline Subclassing**:
   - Create `ZImagePipeline` that properly handles Qwen components
   - Override text encoding methods

3. **Direct Component Usage**:
   - Load all components separately (transformer, vae, tokenizer, text_encoder)
   - Manually implement inference loop

#### Recommendations

- **Immediate Testing**: Test token addition and embedding injection in isolation
- **Fallback Plan**: Prepare custom inference if pipeline modification fails
- **Documentation**: Track findings as this may be the first Z-Image textual inversion implementation
- **Community**: Consider sharing results if successful, as this could benefit the AI community