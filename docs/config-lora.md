# LoRA Training Configuration Reference

Complete reference for Z-Image LoRA training configuration parameters.

## Configuration Structure

```yaml
model:
  name_or_path: string

lora:
  rank: int
  alpha: float
  target_modules: list[string]

dataset:
  path: string
  resolution: int
  center_crop: bool
  random_flip: bool
  num_workers: int

train:
  batch_size: int
  gradient_accumulation_steps: int
  num_epochs: int
  learning_rate: float
  adam_beta1: float
  adam_beta2: float
  adam_epsilon: float
  weight_decay: float
  max_grad_norm: float
  save_every: int

output:
  path: string

logging:
  level: string

dtype: string
mixed_precision: string
seed: int
```

---

## Model Configuration

### `model.name_or_path`
- **Type**: `string`
- **Required**: Yes
- **Description**: Path to the Z-Image model, either a HuggingFace model ID or local directory path
- **Examples**: 
  - `"Tongyi-MAI/Z-Image"` (HuggingFace)
  - `"Tongyi-MAI/Z-Image-Turbo"`
  - `"/path/to/local/model"`
- **Notes**: The model directory must contain `transformer/`, `text_encoder/`, `tokenizer/`, and `vae/` subdirectories

---

## LoRA Configuration

### `lora.rank`
- **Type**: `int`
- **Required**: Yes
- **Default**: `4`
- **Range**: `1` to `128`
- **Suggested**: `4` to `32`
- **Description**: The rank (dimension) of the LoRA decomposition matrices. Higher rank = more parameters = more capacity but slower training
- **Impact**:
  - `rank=4`: ~2MB, fast training, good for style/minor adjustments
  - `rank=8-16`: ~4-8MB, balanced, most common choice
  - `rank=32+`: ~16MB+, high capacity for complex concepts
- **Memory**: Approximately `rank * 2 * hidden_dim * num_target_layers` parameters

### `lora.alpha`
- **Type**: `float`
- **Required**: Yes
- **Default**: `1.0`
- **Range**: `0.1` to `32.0`
- **Suggested**: `rank / 2` to `rank * 2`
- **Description**: Scaling factor for LoRA outputs. The effective learning rate is `alpha / rank`
- **Common Values**:
  - `alpha = rank`: 1:1 scaling (most common)
  - `alpha = rank / 2`: More conservative
  - `alpha = rank * 2`: More aggressive adaptation
- **Notes**: Does not affect model size, only the magnitude of adaptations

### `lora.target_modules`
- **Type**: `list[string]`
- **Required**: Yes
- **Default**: `["to_q", "to_k", "to_v", "to_out"]`
- **Description**: List of module names to apply LoRA to. These are substrings matched against module names in the transformer
- **Common Targets**:
  - `["to_q", "to_k", "to_v", "to_out"]`: All attention projections (recommended)
  - `["to_q", "to_k", "to_v"]`: Query, key, value only
  - `["to_out"]`: Output projection only (minimal)
- **Impact**: More targets = more parameters = higher capacity but slower training
- **Advanced**: Can include MLP layers like `["to_q", "to_k", "to_v", "to_out", "fc1", "fc2"]`

---

## Dataset Configuration

### `dataset.path`
- **Type**: `string`
- **Required**: Yes
- **Description**: Path to your training dataset directory
- **Structure**: Directory containing images (`.jpg`, `.jpeg`, `.png`, `.webp`) with optional `.txt` caption files
- **Example Structure**:
  ```
  dataset/
  ├── image1.jpg
  ├── image1.txt
  ├── image2.png
  ├── image2.txt
  └── ...
  ```

### `dataset.resolution`
- **Type**: `int`
- **Required**: Yes
- **Default**: `512`
- **Common Values**: `256`, `512`, `768`, `1024`
- **Description**: Target resolution for training images (both width and height)
- **Notes**: 
  - Must be divisible by 16 (VAE requirement)
  - Higher resolution = more VRAM usage
  - Z-Image supports various resolutions, but 512 is standard for base model
- **VRAM Impact**:
  - `512`: ~8-12GB
  - `768`: ~16-20GB
  - `1024`: ~24GB+

### `dataset.center_crop`
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Description**: Whether to center-crop images to square aspect ratio
- **Options**:
  - `true`: Crop from center (preserves main subject, may lose edges)
  - `false`: Resize only (may distort aspect ratio)

### `dataset.random_flip`
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Description**: Randomly flip images horizontally during training for data augmentation
- **Recommended**: 
  - `true`: For generic subjects (landscapes, objects)
  - `false`: For text, faces with specific orientation, asymmetric subjects

### `dataset.num_workers`
- **Type**: `int`
- **Required**: No
- **Default**: `4`
- **Range**: `0` to `8`
- **Description**: Number of parallel data loading workers
- **Notes**:
  - `0`: Single-threaded, slower but less memory
  - `4-8`: Faster loading, more CPU/memory usage
  - Set to `0` if you encounter data loading errors

---

## Training Configuration

### `train.batch_size`
- **Type**: `int`
- **Required**: Yes
- **Default**: `1`
- **Range**: `1` to `8`
- **Description**: Number of samples processed simultaneously
- **VRAM Guidelines**:
  - 24GB: `batch_size=1-2`
  - 40GB: `batch_size=2-4`
  - 80GB: `batch_size=4-8`
- **Notes**: Combine with `gradient_accumulation_steps` for effective larger batches

### `train.gradient_accumulation_steps`
- **Type**: `int`
- **Required**: No
- **Default**: `4`
- **Range**: `1` to `32`
- **Description**: Number of forward passes before updating weights. Effective batch size = `batch_size * gradient_accumulation_steps`
- **Recommended**: `4-8` for most cases
- **Example**: `batch_size=1` + `gradient_accumulation_steps=4` = effective batch of 4

### `train.num_epochs`
- **Type**: `int`
- **Required**: Yes
- **Default**: `10`
- **Range**: `5` to `100`
- **Suggested**:
  - Small dataset (<100 images): `20-50 epochs`
  - Medium dataset (100-500 images): `10-20 epochs`
  - Large dataset (500+ images): `5-10 epochs`
- **Description**: Number of complete passes through the dataset
- **Notes**: Monitor for overfitting; save checkpoints to pick best epoch

### `train.learning_rate`
- **Type**: `float`
- **Required**: Yes
- **Default**: `1e-4` (0.0001)
- **Range**: `1e-6` to `1e-3`
- **Suggested**: `5e-5` to `2e-4`
- **Description**: Step size for optimizer updates
- **Guidelines**:
  - `1e-4`: Safe default for most cases
  - `5e-5`: More conservative, slower convergence
  - `2e-4`: Faster training, risk of instability
- **Notes**: Lower for fine-tuning existing LoRAs, higher for training from scratch

### `train.adam_beta1`
- **Type**: `float`
- **Required**: No
- **Default**: `0.9`
- **Range**: `0.8` to `0.95`
- **Description**: Exponential decay rate for first moment estimates (momentum)
- **Notes**: Standard value, rarely needs adjustment

### `train.adam_beta2`
- **Type**: `float`
- **Required**: No
- **Default**: `0.999`
- **Range**: `0.99` to `0.9999`
- **Description**: Exponential decay rate for second moment estimates (variance)
- **Notes**: Standard value, rarely needs adjustment

### `train.adam_epsilon`
- **Type**: `float`
- **Required**: No
- **Default**: `1e-8`
- **Range**: `1e-10` to `1e-6`
- **Description**: Small constant for numerical stability in Adam optimizer
- **Notes**: Standard value, rarely needs adjustment

### `train.weight_decay`
- **Type**: `float`
- **Required**: No
- **Default**: `0.01`
- **Range**: `0.0` to `0.1`
- **Suggested**: `0.01` to `0.05`
- **Description**: L2 regularization penalty to prevent overfitting
- **Guidelines**:
  - `0.0`: No regularization (may overfit)
  - `0.01`: Standard regularization
  - `0.05+`: Stronger regularization for small datasets

### `train.max_grad_norm`
- **Type**: `float`
- **Required**: No
- **Default**: `1.0`
- **Range**: `0.5` to `10.0`
- **Description**: Maximum gradient norm for gradient clipping (prevents exploding gradients)
- **Notes**: `1.0` is standard; increase if training is unstable

### `train.save_every`
- **Type**: `int`
- **Required**: Yes
- **Default**: `500`
- **Range**: `100` to `2000`
- **Description**: Save checkpoint every N training steps
- **Calculation**: `steps_per_epoch = dataset_size / (batch_size * gradient_accumulation_steps)`
- **Recommended**: Save 2-5 times per epoch for small datasets, less frequently for large datasets

---

## Output Configuration

### `output.path`
- **Type**: `string`
- **Required**: Yes
- **Default**: `"./output/zimage_lora"`
- **Description**: Directory where checkpoints and logs will be saved
- **Contents**:
  - `config.yaml`: Copy of training configuration
  - `zimage_lora_step_*.safetensors`: Model checkpoints
  - `zimage_lora_final.safetensors`: Final checkpoint
- **Notes**: Directory will be created automatically if it doesn't exist

---

## Logging Configuration

### `logging.level`
- **Type**: `string`
- **Required**: No
- **Default**: `"INFO"`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Description**: Logging verbosity level
- **Recommended**: `"INFO"` for normal training, `"DEBUG"` for troubleshooting

---

## Global Settings

### `dtype`
- **Type**: `string`
- **Required**: No
- **Default**: `"bfloat16"`
- **Options**: `"float32"`, `"float16"`, `"bfloat16"`
- **Description**: Data type for model weights and computations
- **Recommendations**:
  - `"bfloat16"`: Best for modern GPUs (A100, H100, RTX 40xx)
  - `"float16"`: Good for older GPUs (V100, RTX 30xx)
  - `"float32"`: Highest precision, slowest, highest VRAM
- **Notes**: Z-Image officially uses bfloat16

### `mixed_precision`
- **Type**: `string`
- **Required**: No
- **Default**: `"bf16"`
- **Options**: `"no"`, `"fp16"`, `"bf16"`
- **Description**: Accelerate mixed precision training mode
- **Notes**: Should match `dtype` setting
  - `dtype: "bfloat16"` → `mixed_precision: "bf16"`
  - `dtype: "float16"` → `mixed_precision: "fp16"`

### `seed`
- **Type**: `int`
- **Required**: No
- **Default**: `42`
- **Range**: Any integer
- **Description**: Random seed for reproducibility
- **Notes**: Same seed + same config + same data = same results

---

## Example Configurations

### Minimal (Fast Training, Small Dataset)
```yaml
lora:
  rank: 4
  alpha: 4.0

train:
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 20
  learning_rate: 1e-4
  save_every: 200
```

### Balanced (Recommended Starting Point)
```yaml
lora:
  rank: 8
  alpha: 8.0

train:
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 10
  learning_rate: 1e-4
  save_every: 500
```

### High Capacity (Complex Concepts, Large Dataset)
```yaml
lora:
  rank: 32
  alpha: 16.0
  target_modules:
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out"
    - "fc1"
    - "fc2"

train:
  batch_size: 2
  gradient_accumulation_steps: 2
  num_epochs: 5
  learning_rate: 5e-5
  save_every: 1000
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` to 1
- Reduce `resolution` (e.g., 512 → 384)
- Reduce `lora.rank`
- Reduce `dataset.num_workers`

### Training Too Slow
- Increase `batch_size` if VRAM allows
- Reduce `gradient_accumulation_steps`
- Reduce `dataset.resolution`
- Increase `dataset.num_workers`

### Overfitting (Loss decreases but quality degrades)
- Reduce `num_epochs`
- Increase `weight_decay`
- Add `random_flip: true`
- Use lower `learning_rate`

### Underfitting (Loss plateaus, poor quality)
- Increase `lora.rank`
- Increase `num_epochs`
- Increase `learning_rate`
- Add more training data

---

## Tips & Best Practices

1. **Start Small**: Begin with low rank (4-8) and short training, then scale up
2. **Monitor Checkpoints**: Save frequently and compare outputs from different steps
3. **Effective Batch Size**: Aim for effective batch size of 4-8 (batch_size × gradient_accumulation_steps)
4. **Dataset Quality**: 20-50 high-quality images often better than 500 low-quality ones
5. **Regularization**: For datasets <50 images, use higher weight_decay (0.05+)
6. **Learning Rate**: If loss explodes, halve the learning rate; if loss doesn't decrease, try doubling it
7. **Reproducibility**: Set `seed` for consistent results across runs
