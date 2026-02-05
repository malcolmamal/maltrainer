# Z-Image Textual Inversion ComfyUI Node

This document describes the custom ComfyUI node for Z-Image textual inversion support, enabling visual workflow integration with embeddings trained using maltrainer.

## Overview

The Z-Image Textual Inversion node allows you to use textual inversion embeddings in ComfyUI workflows with Z-Image models. This bridges the gap between training embeddings with maltrainer and using them in visual AI pipelines.

## Features

- **Z-Image Native**: Uses Qwen3 tokenizer and text encoder for authentic Z-Image text encoding
- **Embedding Injection**: Loads and injects safetensors embeddings trained with maltrainer or ai-toolkit
- **Dynamic Loading**: Automatically downloads and caches Z-Image model components
- **ComfyUI Standard**: Returns CONDITIONING output compatible with Z-Image generation nodes
- **Flexible Configuration**: Supports custom model IDs, embedding files, and token strings

## Installation

### Prerequisites
- ComfyUI installed and running
- Python dependencies installed

### Setup Steps

1. **Copy Node Files**:
   ```bash
   # Files are located in maltrainer/comfyui_nodes/
   cp maltrainer/comfyui_nodes/zimage_textual_inversion.py /path/to/ComfyUI/custom_nodes/
   cp maltrainer/comfyui_nodes/__init__.py /path/to/ComfyUI/custom_nodes/
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r maltrainer/comfyui_nodes/requirements.txt
   ```

3. **Restart ComfyUI**:
   Restart your ComfyUI server to load the new node.

## Node Parameters

### Required Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | STRING | Text prompt with embedding token (e.g., "a photo of <trigger>") |

### Optional Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | STRING | "Tongyi-MAI/Z-Image" | HuggingFace model ID for Z-Image |
| `embedding_file` | STRING | "" | Path to safetensors embedding file |
| `token_string` | STRING | "<*>" | Token used in embedding training |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `CONDITIONING` | CONDITIONING | Encoded text embeddings for Z-Image generation |

## Usage Examples

### Basic Z-Image Generation
```
[Z-Image Text Encode (with Embeddings)]
    text: "a beautiful landscape"
    model_id: "Tongyi-MAI/Z-Image"
    ↓
[Z-Image Generation Node]
    conditioning: (from above)
    ↓
[Save Image]
```

### With Textual Inversion
```
[Z-Image Text Encode (with Embeddings)]
    text: "a photo of <sks> in a garden"
    embedding_file: "/path/to/my_embedding.safetensors"
    token_string: "<sks>"
    ↓
[Z-Image Generation Node]
    conditioning: (from above)
    ↓
[Save Image]
```

### Advanced Workflow
```
[Text Input]
    text: "portrait of <character> in <style>"
    ↓
[Z-Image Text Encode (with Embeddings)]
    text: (from above)
    embedding_file: "/models/embeddings/character.safetensors"
    token_string: "<character>"
    ↓
[Z-Image Text Encode (with Embeddings)]
    text: (from above)
    embedding_file: "/models/embeddings/style.safetensors"
    token_string: "<style>"
    ↓
[Z-Image Generation Node]
    conditioning: (from above)
    guidance_scale: 3.5
    num_inference_steps: 20
    ↓
[Image Output]
```

## Embedding File Format

The node expects embeddings in safetensors format with the following structure:

```python
{
    "emb_params": torch.Tensor  # Shape: [num_vectors, 2560]
}
```

### Compatibility
- **Trained with**: maltrainer, ai-toolkit
- **Format**: Safetensors (.safetensors)
- **Dimensions**: 2560 per vector (Qwen3 embedding size)
- **Vectors**: 1-8 vectors supported

### File Size
- **1 vector**: ~10-15KB
- **4 vectors**: ~40-60KB
- **8 vectors**: ~80-100KB

## Technical Details

### Model Loading
- Automatically loads Qwen2Tokenizer and Qwen2ForCausalLM from Z-Image model
- Components cached locally after first use (~20-30GB total)
- GPU acceleration when available

### Text Processing
1. **Tokenization**: Uses Qwen3 tokenizer with custom vocabulary
2. **Embedding Injection**: Replaces token embeddings with trained vectors
3. **Encoding**: Generates text embeddings using Qwen3 transformer
4. **Output**: Returns embeddings in ComfyUI CONDITIONING format

### Memory Usage
- **Base model**: ~20-30GB VRAM for Z-Image components
- **Text encoding**: Minimal additional memory (~100MB)
- **Embeddings**: Negligible memory impact

## Troubleshooting

### Common Issues

**Node not appearing in ComfyUI**
- Ensure files are in `custom_nodes/` directory
- Check ComfyUI logs for import errors
- Restart ComfyUI completely

**Model loading errors**
- Verify internet connection for first-time download
- Check available disk space (~30GB required)
- Ensure CUDA is available for GPU acceleration

**Embedding not working**
- Verify embedding file path is correct
- Check token_string matches training configuration
- Ensure embedding was trained with compatible dimensions

**Generation failures**
- Confirm Z-Image generation node is compatible
- Check conditioning output is properly connected
- Verify model_id matches your Z-Image checkpoint

### Debug Tips

1. **Check Logs**: ComfyUI console shows loading progress and errors
2. **Test Without Embeddings**: Verify basic functionality first
3. **Validate Embeddings**: Use maltrainer's inference script to test embeddings independently

## Integration with maltrainer

### Training Workflow
1. **Train Embeddings**: Use `train_zimage_embedding.py` with YAML config
2. **Export Embeddings**: Automatic safetensors output during training
3. **Use in ComfyUI**: Load embeddings with this node in visual workflows

### File Organization
```
maltrainer/
├── train_zimage_embedding.py
├── config/train_zimage_embedding_example.yaml
├── output/zimage_embedding/final_embedding.safetensors
└── comfyui_nodes/
    ├── zimage_textual_inversion.py
    └── requirements.txt
```

### Best Practices
- Train embeddings with representative dataset
- Use descriptive token strings for multiple embeddings
- Save checkpoints during training for evaluation
- Test embeddings in isolation before complex workflows

## Performance Considerations

### Speed
- **First load**: 2-5 minutes (model download + loading)
- **Subsequent loads**: <30 seconds (cached)
- **Text encoding**: <1 second per prompt

### Optimization Tips
- Use GPU for faster encoding
- Cache models locally to avoid re-downloads
- Pre-load frequently used embeddings
- Use efficient token strings

## Limitations

- **Qwen3 Specific**: Only works with Z-Image (Qwen3-based) models
- **Memory Requirements**: Requires significant VRAM for model loading
- **Single Embedding**: Currently supports one embedding file per node
- **No LoRA**: This node is for textual inversion only (LoRA requires separate handling)

## Future Enhancements

Potential improvements for future versions:
- Support for multiple embedding files
- Integrated LoRA loading
- Batch processing capabilities
- Custom model configurations
- Performance optimizations

## Support

For issues or questions:
1. Check ComfyUI logs for error messages
2. Verify embedding compatibility with maltrainer
3. Test with simple prompts before complex workflows
4. Refer to maltrainer documentation for training guidance

## Changelog

### v1.0.0
- Initial release
- Z-Image Qwen3 text encoding support
- Safetensors embedding loading
- ComfyUI CONDITIONING output format
- Dynamic model loading and caching