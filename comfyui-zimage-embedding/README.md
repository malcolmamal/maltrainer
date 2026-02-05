# Z-Image Textual Inversion ComfyUI Node

This custom ComfyUI node enables textual inversion support for Z-Image models.

## Features

- **Z-Image Compatible**: Uses Qwen3 tokenizer and text encoder for proper Z-Image text encoding
- **Textual Inversion Support**: Load and inject custom embeddings trained with maltrainer or ai-toolkit
- **Dynamic Model Loading**: Automatically loads Z-Image components when needed
- **ComfyUI Integration**: Returns standard CONDITIONING output for use with Z-Image generation nodes

## Installation

1. Copy the `zimage_textual_inversion.py` file to your ComfyUI `custom_nodes/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Restart ComfyUI

## Usage

1. **Add Node**: Search for "Z-Image Text Encode (with Embeddings)" in ComfyUI
2. **Configure**:
   - `text`: Your prompt with embedding token (e.g., "a photo of <trigger>")
   - `model_id`: Z-Image model ID (default: "Tongyi-MAI/Z-Image")
   - `embedding_file`: Path to your .safetensors embedding file (optional)
   - `token_string`: The token used in your embedding (default: "<*>")
3. **Connect**: Link the CONDITIONING output to your Z-Image generation node

## Embedding File Format

The node expects embeddings in safetensors format with:
- Key: `"emb_params"` containing the embedding vectors
- Shape: `[num_vectors, 2560]` where num_vectors is 1-8
- Compatible with embeddings trained using maltrainer or ai-toolkit

## Example Workflow

```
[Z-Image Text Encode (with Embeddings)]
    text: "a beautiful landscape with <sks> style"
    embedding_file: "/path/to/my_embedding.safetensors"
    token_string: "<sks>"
    ↓
[Z-Image Generation Node]
    ↓
[Save Image]
```

## Troubleshooting

- **Model Loading Issues**: Ensure you have enough VRAM (8GB+ recommended)
- **Embedding Not Working**: Check that your token_string matches the one used in training
- **Performance**: First load may take time to download model components

## Dependencies

- torch >= 2.0.0
- transformers >= 4.35.0
- safetensors >= 0.4.0
- accelerate >= 0.25.0