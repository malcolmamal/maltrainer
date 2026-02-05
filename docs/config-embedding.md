# Textual Inversion Training Configuration Reference

Complete reference for Z-Image Textual Inversion (embedding) training configuration parameters.

## Configuration Structure

```yaml
model:
  name_or_path: string

embedding:
  trigger: string
  num_vectors: int
  init_text: string

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

## Embedding Configuration

### `embedding.trigger`
- **Type**: `string`
- **Required**: Yes
- **Default**: `"sks"`
- **Description**: The trigger token/word that will activate your learned embedding
- **Guidelines**:
  - Use unique, uncommon tokens (avoid real words like "cat", "person")
  - Short tokens work best (2-4 characters)
  - Lowercase recommended
  - Avoid special characters
- **Common Choices**:
  - `"sks"`: Classic default from Textual Inversion paper
  - `"ohwx"`: Another common choice
  - `"xyz123"`: Alphanumeric combinations
  - `"mychar"`: Descriptive but unique
- **Usage**: In prompts, use like: `"a photo of sks person"`, `"sks in the style of watercolor"`

### `embedding.num_vectors`
- **Type**: `int`
- **Required**: Yes
- **Default**: `1`
- **Range**: `1` to `75` (practical: `1` to `8`)
- **Suggested**: `1` to `4`
- **Description**: Number of embedding vectors to learn for the concept
- **Impact**:
  - `1 vector`: ~2-4KB, simple concepts (single object, simple style)
  - `2-3 vectors`: ~4-8KB, moderate complexity (character with variations, art style)
  - `4-8 vectors`: ~8-16KB, complex concepts (scene composition, detailed character)
  - `8+ vectors`: Rarely needed, may overfit
- **Notes**: 
  - More vectors = more capacity but higher risk of overfitting
  - For Z-Image (Qwen3 encoder): each vector is 2560 dimensions
  - Total size = `num_vectors × 2560 × 4 bytes` (for float32)

### `embedding.init_text`
- **Type**: `string`
- **Required**: Yes
- **Default**: `"a photo"`
- **Description**: Text used to initialize the embedding vectors (starting point for learning)
- **Guidelines**: Choose text semantically close to your concept
- **Examples by Use Case**:
  - **Person**: `"a photo of a person"`, `"a man"`, `"a woman"`
  - **Object**: `"a photo of an object"`, `"a toy"`, `"a sculpture"`
  - **Art Style**: `"a painting"`, `"an artwork"`, `"digital art"`
  - **Animal**: `"a photo of an animal"`, `"a dog"`, `"a cat"`
  - **Scene**: `"a landscape"`, `"an interior room"`
- **Impact**: Better initialization = faster convergence and better results
- **Notes**: The embedding will start from these token embeddings and adapt during training

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
  ├── subject_001.jpg
  ├── subject_001.txt  # Optional: "in a garden"
  ├── subject_002.jpg
  ├── subject_002.txt  # Optional: "wearing a hat"
  └── ...
  ```
- **Notes**: 
  - Captions are optional; trigger token will be automatically prepended
  - For textual inversion, 10-50 images is typical
  - Quality > Quantity: Better to have 20 diverse, high-quality images than 200 similar ones

### `dataset.resolution`
- **Type**: `int`
- **Required**: Yes
- **Default**: `512`
- **Common Values**: `512`, `768`, `1024`
- **Description**: Target resolution for training images (both width and height)
- **Recommendations**:
  - `512`: Standard, fastest training, lowest VRAM (~10-12GB)
  - `768`: Better detail, slower, more VRAM (~16-20GB)
  - `1024`: Highest quality, slowest, most VRAM (~24GB+)
- **Notes**: Must be divisible by 16 (VAE requirement)

### `dataset.center_crop`
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Description**: Whether to center-crop images to square aspect ratio
- **Options**:
  - `true`: Crop from center (recommended for most cases)
  - `false`: Resize/stretch (may distort subjects)
- **Recommendation**: Keep `true` unless your images are already square

### `dataset.random_flip`
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Description**: Randomly flip images horizontally during training
- **Recommendations**:
  - `true`: For symmetric subjects (objects, animals, generic scenes)
  - `false`: For faces, text, asymmetric subjects, specific poses
- **Notes**: Effectively doubles your dataset size by augmentation

### `dataset.num_workers`
- **Type**: `int`
- **Required**: No
- **Default**: `4`
- **Range**: `0` to `8`
- **Description**: Number of parallel data loading workers
- **Recommendations**:
  - `0`: Use if encountering data loading errors, slower
  - `2-4`: Good balance for most systems
  - `8`: Maximum for fast CPUs with plenty of RAM

---

## Training Configuration

### `train.batch_size`
- **Type**: `int`
- **Required**: Yes
- **Default**: `1`
- **Range**: `1` to `4`
- **Description**: Number of samples processed simultaneously
- **Recommendations**:
  - For textual inversion, `batch_size=1` is most common
  - Larger batches may help with stability but require more VRAM
- **VRAM Guidelines** (at 512px):
  - 12GB: `batch_size=1`
  - 24GB: `batch_size=1-2`
  - 40GB+: `batch_size=2-4`

### `train.gradient_accumulation_steps`
- **Type**: `int`
- **Required**: No
- **Default**: `4`
- **Range**: `1` to `16`
- **Description**: Number of forward passes before updating embeddings. Effective batch size = `batch_size * gradient_accumulation_steps`
- **Recommended**: `4-8` for textual inversion
- **Notes**: Helps stabilize training when batch_size=1

### `train.num_epochs`
- **Type**: `int`
- **Required**: Yes
- **Default**: `100`
- **Range**: `50` to `500`
- **Suggested**:
  - Tiny dataset (5-10 images): `200-500 epochs`
  - Small dataset (10-30 images): `100-200 epochs`
  - Medium dataset (30-50 images): `50-100 epochs`
  - Large dataset (50+ images): `30-50 epochs`
- **Description**: Number of complete passes through the dataset
- **Notes**: 
  - Textual inversion typically needs many more epochs than LoRA
  - Watch for overfitting after loss plateaus
  - Use `save_every` to evaluate checkpoints at different stages

### `train.learning_rate`
- **Type**: `float`
- **Required**: Yes
- **Default**: `5e-4` (0.0005)
- **Range**: `1e-4` to `1e-3`
- **Suggested**: `3e-4` to `5e-4`
- **Description**: Step size for embedding updates
- **Guidelines**:
  - `5e-4`: Good default for most cases
  - `3e-4`: More conservative, slower but stable
  - `1e-3`: Aggressive, faster but may be unstable
  - `1e-4`: Very conservative, use if training diverges
- **Notes**: 
  - Textual inversion uses higher LR than LoRA training
  - Higher LR needed because only embedding table is updated (small parameter count)

### `train.adam_beta1`
- **Type**: `float`
- **Required**: No
- **Default**: `0.9`
- **Range**: `0.8` to `0.95`
- **Description**: Adam optimizer momentum parameter
- **Notes**: Standard value works well, rarely needs tuning

### `train.adam_beta2`
- **Type**: `float`
- **Required**: No
- **Default**: `0.999`
- **Range**: `0.99` to `0.9999`
- **Description**: Adam optimizer second moment parameter
- **Notes**: Standard value works well, rarely needs tuning

### `train.adam_epsilon`
- **Type**: `float`
- **Required**: No
- **Default**: `1e-8`
- **Range**: `1e-10` to `1e-6`
- **Description**: Small constant for numerical stability
- **Notes**: Standard value, no need to change

### `train.weight_decay`
- **Type**: `float`
- **Required**: No
- **Default**: `0.01`
- **Range**: `0.0` to `0.1`
- **Suggested**: `0.01` to `0.05`
- **Description**: L2 regularization to prevent overfitting
- **Guidelines**:
  - `0.0`: No regularization (may overfit on small datasets)
  - `0.01`: Standard regularization
  - `0.05-0.1`: Strong regularization for very small datasets (<10 images)
- **Notes**: Important for textual inversion due to small parameter count

### `train.max_grad_norm`
- **Type**: `float`
- **Required**: No
- **Default**: `1.0`
- **Range**: `0.5` to `2.0`
- **Description**: Maximum gradient norm for clipping
- **Notes**: Prevents gradient explosions; standard value works well

### `train.save_every`
- **Type**: `int`
- **Required**: Yes
- **Default**: `500`
- **Range**: `100` to `1000`
- **Description**: Save checkpoint every N training steps
- **Recommendations**:
  - Small dataset: Save every 100-200 steps (every 1-2 epochs)
  - Medium dataset: Save every 500 steps
  - Large dataset: Save every 1000 steps
- **Calculation**: `steps_per_epoch = dataset_size / (batch_size * gradient_accumulation_steps)`
- **Notes**: Textual inversion can plateau/overfit, so save checkpoints frequently to evaluate different training stages

---

## Output Configuration

### `output.path`
- **Type**: `string`
- **Required**: Yes
- **Default**: `"./output/zimage_embedding"`
- **Description**: Directory where embeddings and logs will be saved
- **Contents**:
  - `config.yaml`: Copy of training configuration
  - `zimage_embedding_step_*.safetensors`: Embedding checkpoints
  - `zimage_embedding_final.safetensors`: Final embedding
- **File Size**: ~10-100KB per embedding (depending on `num_vectors`)
- **Notes**: Directory created automatically if it doesn't exist

---

## Logging Configuration

### `logging.level`
- **Type**: `string`
- **Required**: No
- **Default**: `"INFO"`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Description**: Logging verbosity
- **Recommended**: 
  - `"INFO"`: Normal training
  - `"DEBUG"`: Troubleshooting or detailed progress

---

## Global Settings

### `dtype`
- **Type**: `string`
- **Required**: No
- **Default**: `"bfloat16"`
- **Options**: `"float32"`, `"float16"`, `"bfloat16"`
- **Description**: Data type for model computations
- **Recommendations**:
  - `"bfloat16"`: Best for modern GPUs (A100, H100, RTX 40xx)
  - `"float16"`: Good for older GPUs (V100, RTX 30xx)
  - `"float32"`: Highest precision, rarely needed
- **Notes**: Embedding vectors are stored at full precision regardless

### `mixed_precision`
- **Type**: `string`
- **Required**: No
- **Default**: `"bf16"`
- **Options**: `"no"`, `"fp16"`, `"bf16"`
- **Description**: Accelerate mixed precision mode
- **Notes**: Should match `dtype`

### `seed`
- **Type**: `int`
- **Required**: No
- **Default**: `42`
- **Range**: Any integer
- **Description**: Random seed for reproducibility
- **Notes**: Same seed = same initialization and training order

---

## Example Configurations

### Simple Person/Character (Single Concept)
```yaml
embedding:
  trigger: "sks"
  num_vectors: 1
  init_text: "a photo of a person"

dataset:
  path: "./dataset/my_character"
  resolution: 512
  random_flip: false  # Keep poses/features consistent

train:
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 100
  learning_rate: 5e-4
  save_every: 200
```

### Art Style (Multi-Vector)
```yaml
embedding:
  trigger: "ohwx"
  num_vectors: 3
  init_text: "a painting in the style of"

dataset:
  path: "./dataset/art_style"
  resolution: 512
  random_flip: true  # More variation

train:
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 150
  learning_rate: 3e-4
  weight_decay: 0.05
  save_every: 300
```

### Complex Subject (High Detail)
```yaml
embedding:
  trigger: "myobj"
  num_vectors: 4
  init_text: "a photo of an intricate object"

dataset:
  path: "./dataset/detailed_object"
  resolution: 768
  center_crop: true
  random_flip: false

train:
  batch_size: 1
  gradient_accumulation_steps: 8
  num_epochs: 200
  learning_rate: 2e-4
  weight_decay: 0.03
  save_every: 500
```

---

## Troubleshooting

### Loss Not Decreasing
- **Increase** `learning_rate` (try 5e-4 → 1e-3)
- **Increase** `num_epochs`
- **Reduce** `weight_decay`
- Check that images load correctly
- Verify `trigger` token doesn't already exist in tokenizer

### Overfitting (Loss Low, Poor Generation Quality)
- **Reduce** `num_epochs` (try earlier checkpoints)
- **Increase** `weight_decay` (0.01 → 0.05)
- **Add** more diverse training images
- **Reduce** `num_vectors`

### Blurry/Generic Results
- **Increase** `num_vectors` (1 → 2-3)
- **Increase** `num_epochs`
- Improve training data quality/diversity
- Try different `init_text` closer to your concept

### Out of Memory
- **Reduce** `resolution` (768 → 512)
- **Reduce** `batch_size` to 1
- **Reduce** `gradient_accumulation_steps`
- **Reduce** `num_workers` to 0

### Training Too Slow
- **Reduce** `num_epochs` initially, add more if needed
- **Reduce** `resolution` (768 → 512)
- **Increase** `batch_size` if VRAM allows

---

## Dataset Guidelines

### Number of Images
- **Minimum**: 5-10 images (may work, but limited variation)
- **Recommended**: 15-30 images for single subjects
- **Optimal**: 30-50 diverse images
- **Maximum Useful**: ~100 images (more doesn't always help)

### Image Quality
- **Resolution**: Higher quality source images (1024px+) resize better
- **Diversity**: Vary poses, angles, lighting, backgrounds
- **Consistency**: Subject should be clearly visible and similar across images
- **Avoid**: Watermarks, heavy filters, extreme crops

### Caption Guidelines
- **Optional**: Captions not required (trigger token auto-added)
- **If Used**: Describe variations/context: `"wearing glasses"`, `"in sunlight"`, `"from the side"`
- **Keep Short**: 5-10 words max per caption
- **Don't Include**: The trigger token (it's added automatically)

---

## Tips & Best Practices

1. **Start with 1 Vector**: Train with `num_vectors=1` first, increase only if needed
2. **Monitor Early Checkpoints**: Best results often at 50-75% of training, not final epoch
3. **Test Regularly**: Generate samples at different checkpoints to find sweet spot
4. **Use Descriptive Init Text**: Better initialization = faster/better convergence
5. **Dataset Quality**: 20 good diverse images > 100 similar/poor images
6. **Learning Rate**: If loss spikes/NaN, halve LR; if barely moves, double it
7. **Unique Trigger**: Avoid common words; check that token doesn't exist: 
   ```python
   tokenizer.encode("sks")  # Should add new token
   ```
8. **Regularization**: Smaller datasets need higher `weight_decay` (0.05+)
9. **Patience**: Textual inversion needs many epochs; don't stop too early
10. **Compare**: Train multiple embeddings with different settings, compare results

---

## Using Trained Embeddings

After training, use your embedding in prompts by including the trigger token:

```
"a photo of sks person in a garden"
"sks wearing a red hat"
"portrait of sks, high quality"
"sks in the style of van gogh"
```

The embedding file (`.safetensors`) can be loaded in:
- ComfyUI (place in `models/embeddings/`)
- Stable Diffusion WebUI (place in `embeddings/`)
- Any Z-Image inference pipeline that supports textual inversion

**File Size**: ~2-50KB depending on `num_vectors`
