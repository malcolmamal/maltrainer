import logging

import torch
from diffusers import ZImagePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load(lora_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image", torch_dtype=torch.bfloat16).to(device)

    logger.info(f"Loading LoRA from {lora_path}")

    # Try loading with diffusers
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        logger.info("LoRA loaded and fused successfully.")
    except Exception:
        logger.exception("Failed to load LoRA")
        return

    logger.info(f"Generating with prompt: {prompt}")
    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save("test_my_lora.png")
    logger.info("Saved test_my_lora.png")


if __name__ == "__main__":
    lora_path = r"c:\Development\trainers\maltrainer\output\zimage_lora\zimage_lora_step_final.safetensors"
    # Use the trigger word from config
    prompt = "sks woman, photo of a woman, high quality"
    test_load(lora_path, prompt)
