#!/usr/bin/env python3
"""
Simple inference script for Z-Image model with optional LoRA and Textual Inversion support.

Usage examples:
1. Basic Z-Image generation:
   python scripts/infer_zimage.py --prompt "a cat in a hat"

2. With LoRA:
   python scripts/infer_zimage.py --prompt "a cat in a hat" --lora_path path/to/lora.safetensors

3. With embedding:
   python scripts/infer_zimage.py --prompt "a <trigger> cat in a hat" --embedding_path path/to/embedding.safetensors

4. With both:
   python scripts/infer_zimage.py --prompt "a <trigger> cat in a hat" --lora_path path/to/lora.safetensors \
       --embedding_path path/to/embedding.safetensors
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import safetensors.torch as safetensors
import torch
from diffusers import ZImagePipeline
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_textual_inversion_embedding(
    pipeline: Any, embedding_path: str, token_string: str = "<*>", model_id: str = "Tongyi-MAI/Z-Image"
) -> None:
    """Load textual inversion embedding and add to tokenizer/text_encoder."""
    logger.info(f"Loading textual inversion from {embedding_path}")

    # Load the embedding
    embedding_data = safetensors.load_file(embedding_path)

    # Assume the embedding is stored under 'emb_params' or similar
    # Adjust based on how we save it in training
    if "emb_params" in embedding_data:
        embedding_vectors = embedding_data["emb_params"]
    elif "embedding" in embedding_data:
        embedding_vectors = embedding_data["embedding"]
    else:
        # Try to find the embedding tensor
        embedding_vectors = None
        for _key, tensor in embedding_data.items():
            if tensor.dim() == 2 and tensor.shape[0] <= 8:  # num_vectors x dim
                embedding_vectors = tensor
                break

    if embedding_vectors is None:
        raise ValueError(f"Could not find embedding vectors in {embedding_path}")

    # For Z-Image, we need to load Qwen components separately since the pipeline might not expose them properly
    tokenizer = Qwen2Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder: Any = Qwen3ForCausalLM.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(pipeline.device)

    # One token per vector, named as the trainer names them: the trigger, then trigger_1, trigger_2, ...
    tokens = [token_string, *(f"{token_string}_{i}" for i in range(1, embedding_vectors.shape[0]))]
    new = [token for token in tokens if token not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_tokens(new)
        text_encoder.resize_token_embeddings(len(tokenizer))

    # Set each token's row
    table = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        for token, vector in zip(tokens, embedding_vectors, strict=True):
            table[tokenizer.convert_tokens_to_ids(token)] = vector.to(text_encoder.device, text_encoder.dtype)

    logger.info(f"Loaded {embedding_vectors.shape[0]} embedding vectors for token '{token_string}'")

    # Replace pipeline's text_encoder and tokenizer with our modified ones
    # Note: This assumes the pipeline allows attribute replacement
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = text_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-Image inference with LoRA and Textual Inversion support")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save generated image")
    parser.add_argument("--model_id", type=str, default="Tongyi-MAI/Z-Image", help="HuggingFace model ID")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA weights (.safetensors)")
    parser.add_argument("--embedding_path", type=str, help="Path to textual inversion embedding (.safetensors)")
    parser.add_argument("--token_string", type=str, default="<*>", help="Token string for textual inversion")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")

    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load pipeline
    logger.info(f"Loading Z-Image pipeline from {args.model_id}")
    pipeline = cast(Any, ZImagePipeline).from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipeline = pipeline.to(device)

    # Load LoRA if specified
    if args.lora_path:
        logger.info(f"Loading LoRA from {args.lora_path}")
        pipeline.load_lora_weights(args.lora_path)
        pipeline.fuse_lora()

    # Load textual inversion if specified
    if args.embedding_path:
        load_textual_inversion_embedding(pipeline, args.embedding_path, args.token_string, args.model_id)

    # Generate images
    output_path = Path(args.output_path)
    output_dir, base, ext = output_path.parent, output_path.stem, output_path.suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ext:
        ext = ".png"

    # Use provided seed or generate random one
    initial_seed = args.seed if args.seed is not None else int(torch.randint(0, 2**32, (1,)).item())

    for i in range(args.num_images):
        current_seed = initial_seed + i
        generator = torch.Generator(device=device).manual_seed(current_seed)

        logger.info(f"Generating image {i + 1}/{args.num_images} with seed {current_seed}")

        with torch.no_grad():
            image = pipeline(
                prompt=args.prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                width=args.width,
                height=args.height,
                generator=generator,
            ).images[0]

        # Save image with timestamp and index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.num_images > 1:
            final_output_path = output_dir / f"{base}_{timestamp}_{i + 1}{ext}"
        else:
            final_output_path = output_dir / f"{base}_{timestamp}{ext}"

        image.save(final_output_path)
        logger.info(f"Image saved to {final_output_path}")


if __name__ == "__main__":
    main()
