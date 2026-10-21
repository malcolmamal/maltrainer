# Z-Image Textual Inversion (Embeddings)

> **Deprecated.** Z-Image textual inversion was a proof of concept that did not give usable results. The trainer
> (`train_zimage_embedding.py`) still runs, is tested and logs a deprecation warning when started, and the approach
> may be reused with other models later. For Z-Image, train a LoRA instead.

## Overview
Textual Inversion (TI) is a technique for teaching a model new concepts (objects, styles, faces) by finding a new "word" in the text encoder's vocabulary that represents that concept. Unlike LoRA, which modifies the model's *processing* weights, TI only optimizes the *input* vector for a specific token.

Z-Image uses a Qwen2 LLM as its text encoder, so the hope was that its rich embedding space would make a single trained vector effective. In practice it did not: the vectors this trainer produced did not carry a usable likeness or concept, which is why the trainer is deprecated. The rest of this page describes what the trainer does, for anyone reusing it.

## How it Works
1.  **Architecture**: Z-Image typically uses a Transformer for image generation and Qwen2 for text encoding.
2.  **Process**:
    *   We introduce a new placeholder token (e.g., `anya` or `<sks>`) into the Qwen2 tokenizer.
    *   This token is assigned a random initialization (e.g., the vector for "woman" or "photo").
    *   During training, we **freeze** the entire Z-Image Transformer and the Qwen2 Text Encoder weights.
    *   We **only** update the vector weights associated with our new token.
    *   The goal is to move this vector in the embedding space until it minimizes the error when reconstructing your training images.

## What to Expect
*   **File Size**: Extremely small (typically < 10KB). It is literally just one (or a few) vectors of numbers.
*   **Training Speed**: Very fast compared to LoRA or fine-tuning, as there are very few parameters to optimize.
*   **Flexibility**: You can mix and match multiple embeddings in a prompt (e.g., "a photo of <emb1> in the style of <emb2>").

## LoRA vs. Textual Inversion

| Feature | LoRA (Low-Rank Adaptation) | Textual Inversion (Embedding) |
| :--- | :--- | :--- |
| **Mechanism** | Injects low-rank matrices into the model's attention/linear layers. Effectively "brain surgery" on the model. | Finds a specific coordinate in the text encoder's "dictionary" that represents the concept. Does not touch the model's brain. |
| **File Size** | Medium to Large (100MB - 2GB) | Tiny (4KB - 20KB) |
| **Likeness/Fidelity** | **High**. Can teach the model completely new compositions, distinct features, and complex details. | **Medium**. Good for styles and general object characteristics. May struggle with very specific facial details compared to LoRA. |
| **Training Time** | Slower (requires backprop through deep layers). | Faster (vectors are shallow). |
| **Compute Req** | Higher VRAM (requires storing gradients for model weights). | Lower VRAM (gradients only for the input vector). |
| **Concept Bleed** | Can sometimes "overcook" the model or bleed into other concepts if not regularized. | minimal. The model itself remains unchanged. |

### Recommendation
*   **Use LoRA** for Z-Image: character likeness, styles and objects (see [zimage_lora_training.md](zimage_lora_training.md)).
*   The embedding trainer is kept only as a starting point for other models; the small file size is its one real advantage, and on Z-Image it did not pay for the missing fidelity.

## Usage in Maltrainer

### Training
Use the `train_zimage_embedding.py` script.
```bash
python train_zimage_embedding.py --config config/myembedding.yaml
```

### Inference (CLI)
Use the `scripts/infer_zimage.py` script.
```bash
python scripts/infer_zimage.py --prompt "a photo of anya" --embedding_path output/zimage_embedding_anya/zimage_embedding_final.safetensors --token_string "anya"
```

### Inference (ComfyUI)
We provide a custom node in `comfyui-zimage-embedding`.
1.  Copy the folder to `ComfyUI/custom_nodes/`.
2.  Use the `Z-Image Text Encode (with Embeddings)` node.
3.  Point `embedding_file` to your `.safetensors` file.
4.  Set `token_string` to the trigger word you used (e.g., `anya`).

**Note on ComfyUI**: The current implementation loads a separate Qwen2 instance. This ensures compatibility but requires VRAM for the text encoder.

## Architecture Note
Z-Image uses **Qwen3** for its text encoder. The training and inference scripts have been updated to use Qwen3ForCausalLM.
