# Maltrainer

Modern AI model training toolkit focused on Z-Image and modern diffusion architectures.

## Features

- **Z-Image LoRA Training**: Train LoRA adapters for Z-Image (base) models
- **Z-Image Textual Inversion**: Train textual inversion embeddings for Z-Image
- **YAML Configuration**: Simple YAML-based configuration system
- **Modern Architecture**: Clean, modular codebase built on PyTorch and Diffusers

## Installation

### 1. Install PyTorch with CUDA support

For GPU training on Windows, you **must** install the correct CUDA version of PyTorch for your system before installing the rest of the requirements.

**For NVIDIA RTX 50-series (Blackwell) / CUDA 13.0:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**For other modern GPUs (CUDA 12.4 or 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# OR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install remaining requirements

```bash
pip install -r requirements.txt
```

## Web UI (Trainer Frontend)

The trainer UI is an Angular app that talks to a FastAPI backend over HTTP + WebSocket.

### Backend setup

1. Set datasets root (optional). Defaults to ./datasets when not set:
   ```bash
   # Windows (PowerShell)
   $env:DATASETS_PATH = "C:\Development\projects\maltrainer\datasets"
   # Linux/macOS
   export DATASETS_PATH="/path/to/datasets"
   ```

2. Run the API server:
   ```bash
   python -m uvicorn backend.app:app --host 0.0.0.0 --port 8008
   ```

### Frontend setup

1. Install UI dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Run the UI:
   ```bash
   npm start
   ```

3. Open http://localhost:3003

The UI will list datasets from DATASETS_PATH, let you start training or dry runs, and show live logs/progress.

## Quick Start

### LoRA Training

1. Prepare your dataset (images with optional `.txt` caption files)
2. Copy and edit the example config:
   ```bash
   cp config/train_zimage_lora_example.yaml config/my_lora.yaml
   ```
3. Run training:
   ```bash
   python train_zimage_lora.py --config config/my_lora.yaml
   ```

### Textual Inversion Training

1. Prepare your dataset (images with optional `.txt` caption files)
2. Copy and edit the example config:
   ```bash
   cp config/train_zimage_embedding_example.yaml config/my_embedding.yaml
   ```
3. Run training:
   ```bash
   python train_zimage_embedding.py --config config/my_embedding.yaml
   ```

## Training

### Preparing Your Dataset

Before training, organize your images in a directory structure like this:

```
my_dataset/
├── image1.jpg
├── image1.txt    # Optional caption file
├── image2.png
├── image2.txt
└── ...
```

- **Images**: Supported formats include `.jpg`, `.jpeg`, `.png`, `.webp`
- **Captions**: Text files with the same name as the image (optional)
- **Resolution**: Images will be resized according to your config (default 512x512)

### Configuring Training

1. **Choose the right config template**:
   - For LoRA training: `config/train_zimage_lora_example.yaml`
   - For Textual Inversion: `config/train_zimage_embedding_example.yaml`

2. **Edit the configuration**:
   - Set `dataset.path` to your dataset directory
   - Adjust `train.batch_size` and `gradient_accumulation_steps` based on your GPU VRAM
   - Modify learning rates and epochs as needed
   - Configure output paths and logging

3. **Key parameters to customize**:
   - `model.name_or_path`: Path to Z-Image model
   - `dataset.resolution`: Training resolution
   - `train.learning_rate`: Learning rate (higher for embeddings)
   - `train.num_epochs`: Number of training epochs
   - `output.path`: Where to save checkpoints

### Z-Image Turbo Templates

Two Turbo templates are included so you can compare the de-distilled model vs the training adapter approach:

- De-distilled (OneTrainer-style): [config/zimage_turbo_template-onetrainer.yaml](config/zimage_turbo_template-onetrainer.yaml)
- Training adapter (AI Toolkit-style): [config/zimage_turbo_template-adapter.yaml](config/zimage_turbo_template-adapter.yaml)

Notes:
- `model.extras_name_or_path` points the text encoder + VAE to the Turbo base while the transformer loads from the de-distilled model.
- `model.assistant_lora_path` merges the Turbo training adapter into the transformer before training.
- Both templates default to LoRA `rank: 16` and `alpha: 1` to match the OneTrainer config you provided; adjust if you prefer a different rank/alpha.
- Maltrainer now supports logit-normal timestep sampling and LR warmup via `train.timestep_distribution` and `train.learning_rate_warmup_steps`.
- Quantized transformer weights (INT_W8A8) and async offloading/compile are still not implemented, so results may differ from OneTrainer in those areas.

### Running Training

Execute the training script with your config:

```bash
# For LoRA training
python train_zimage_lora.py --config config/my_lora.yaml

# For Textual Inversion
python train_zimage_embedding.py --config config/my_embedding.yaml
```

For testing, you can simulate training without downloading the model:

```bash
# Dry run (default 0.025s per step)
python train_zimage_lora.py --config config/my_lora.yaml --dry-run

# Dry run with custom step time
python train_zimage_embedding.py --config config/my_embedding.yaml --dry-run --dry-run-step-time 0.1
```

### Monitoring Training

- Training progress is logged to the console
- Checkpoints are saved periodically (every 500 steps by default)
- Monitor GPU usage and adjust batch size if needed
- Training can be interrupted and resumed (checkpoints include optimizer state)

### Output Files

Training produces:
- `zimage_lora_step_XXXX.safetensors` or `zimage_embedding_step_XXXX.safetensors`: Model checkpoints
- `config.yaml`: Saved configuration for reference
- Logs in the output directory

## Configuration

All training parameters are configured via YAML files. See `config/` directory for examples.

### Key Configuration Sections

- **model**: Model path and architecture settings
- **lora/embedding**: Adapter-specific configuration
- **dataset**: Dataset path and preprocessing
- **train**: Training hyperparameters
- **output**: Output directory configuration
- **logging**: Logging settings

## Dataset Structure

```
dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

- Images: `.jpg`, `.jpeg`, `.png`, `.webp`
- Captions: Optional `.txt` files with same name as image
- For textual inversion, captions can be empty (trigger token will be used)

## Output Structure

```
output/
├── config.yaml           # Saved configuration
├── zimage_lora_step_500.safetensors
├── zimage_lora_step_1000.safetensors
└── ...
```

## Requirements

- Python 3.10+ (3.12 recommended for RTX 50-series)
- PyTorch 2.10+ (for CUDA 13.0 support)
- CUDA 11.8 - 13.0 (for GPU training)
- 24GB+ VRAM recommended for Z-Image

## Documentation

See `docs/` directory for detailed documentation:
- [Textual Inversion and Z-Image Support](docs/textual-inversion-z-image.md)
- [Training Options and OneTrainer Parity](docs/training-options-and-parity.md)
- [OneTrainer Architecture and Multi-Model Plan](docs/onetrainer-architecture-and-multi-model-plan.md)

## License

MIT License

## Credits

Built with inspiration from:
- ai-toolkit
- musubi-tuner
- kohya_ss
- stable-diffusion-webui
- ComfyUI
