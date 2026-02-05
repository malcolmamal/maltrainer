# Textual Inversion and Z-Image Support

## Background
- Textual inversion (TI) embeddings are the tiny files used in the SD1.5 era: they don’t contain new weights for the UNet or text encoder, just a learned vector that lives within the text encoder embedding table. When the token is inserted into a prompt, the model uses that vector instead of a CLIP token id, so the embedding works like a lightweight concept replacement.
- The vector looks like a single high-dimensional tensor (e.g., 768 dimensions for SD1.5) and is serialized either as safetensors or PyTorch (`.pt`/`.bin`) files that mostly hold one entry such as `string_to_param = {'*': tensor(...)}`.

## ai-toolkit implementation
- `ai-toolkit` already implements this via the `Embedding` helper (`toolkit/embedding.py`). It adds placeholder tokens when training, copies initializer token embeddings, and exposes `vec` getters/setters. During training, `BaseSDTrainProcess` appends the embedding’s parameters to the optimizer if an `embed_config` is supplied, so the same loop that trains LoRA or adapters also updates the TI vector.
- When saving, the vector gets pushed into the text encoder weights and exported as metadata. On inference, the toolkit replaces `[name]`/`[trigger]` or any custom placeholder token when the prompt is encoded, so no extra checkpoint is needed—just the small embedding file.
- Because the Z-Image trainer (`extensions_built_in/diffusion_models/z_image/z_image.py`) already sets up a tokenizer + Qwen3 text encoder, the `Embedding` workflow can operate on that model as well. The TI vector lives in the same text encoder used by the transformer, which means you can train a TI embedding while training Z-Image models or even independently, then reuse it at inference like before.

## Inference tools
- Stable-diffusion-webui (`modules/textual_inversion/textual_inversion.py`) loads these embedding files, keeps the `string_to_param` entry, and exposes the placeholder token through the prompt parser. The same code is why typing `<embedding>` works in SDWebUI when Z-Image isn’t even involved—it just swaps the token for the stored vector before the tokenizer sees the prompt.
- ComfyUI implements a similar flow inside `comfy/sd1_clip.py`: `tokenize_with_weights` scans the prompt for embedding identifiers, `load_embed` fetches the safetensors/pt file, and the tokenizer inserts the vector instead of CLIP IDs. That code also handles bundled embeddings and safetensors metadata, so it mirrors how SDWebUI injects the vector on-the-fly.
- Since both inference stacks (ComfyUI and SDWebUI) expect embeddings to be a lightweight vector that sits next to the tokenizer, you can reuse the same saved file produced by ai-toolkit’s `Embedding` when running the Z-Image pipeline in either tool.

## Additional Ecosystem Notes
- The musubi trainer you added (`musubi-tuner/src/musubi_tuner/zimage_train.py`) already targets the DiT/Z-Image architecture with its own scheduler helpers, so the same TI approach can be ported there—just ensure it writes the `string_to_param` metadata or safetensors tensors like ai-toolkit does.
- At present, kohya_ss’s `train_textual_inversion.py` is strictly for CLIP-based SD1.5/2.1 models and does not support the Qwen3/transformer layout that Z-Image uses. You cannot reuse those embeddings with Z-Image unless the script is updated to load the Qwen3 text encoder and access the transformer embedding table directly.

## Summary
1. ai-toolkit already supports the small TI files for any model whose text encoder is exposed, including Z-Image.
2. Inference tools such as Stable-diffusion-webui and ComfyUI insert TI vectors by detecting the placeholder token when tokenizing prompts, so the same files work across models as long as the tokenizer shares the token id.
3. Musubi and other Z-Image trainers can borrow ai-toolkit’s embedding logic or its serialization format to make TI usable there as well.
4. Kohya_ss/TI remains tied to CLIP/SD1.5 and cannot natively produce Z-Image embeddings without a rewrite of its text encoder handling.
