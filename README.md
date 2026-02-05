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

## License

MIT License

## Credits

Built with inspiration from:
- ai-toolkit
- musubi-tuner
- kohya_ss
- stable-diffusion-webui
- ComfyUI
