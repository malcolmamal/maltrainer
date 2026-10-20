#!/usr/bin/env python3
"""
FLUX.3 [klein] LoRA training for maltrainer.

Trains the base (undistilled) Klein on ComfyUI's single files: the DiT
(flux3_image_klein_int8_convrot, or _bf16), the Flux2 VAE and the Qwen3-VL-4B text encoder
(qwen3vl_4b_int8_convrot, or _bf16). Latents and caption embeddings are cached once, then the
encoders are unloaded before the DiT loads. The prompt upsampler is an inference-time tool and
is not used.

Conventions follow ComfyUI's FLUX.3 (comfy/ldm/flux3, ModelType.FLUX) and the diffusers FLUX.3
trainer. The DiT gets sigma in [0, 1] and predicts v = noise - x0. Sigmas come from the
reference schedule's fixed shift of 6.93, with no resolution-dependent shift.

    python train_flux3_klein_lora.py --config templates/flux3_klein_template.yaml
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.models.flux3.loading import load_flux3_transformer
from maltrainer.models.flux3.lora import Flux3LoRANetwork
from maltrainer.models.flux3.text_encoder import load_flux3_text_encoder
from maltrainer.models.flux3.vae import load_flux2_vae_encoder
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.image_lora import ImageLoRATrainer, free_memory
from maltrainer.training.sigma import LOGIT_NORMAL_NAMES, clamped, logit_normal, shifted
from maltrainer.utils.config import load_config

logger = logging.getLogger(__name__)

# The FLUX.3 reference sampler's shift (ComfyUI sampling_settings mu = ln 6.93, the diffusers
# pipeline's Flux3SetTimestepsStep default). The same at every resolution.
FLUX3_SHIFT = 6.93


class Flux3KleinLoRATrainer(ImageLoRATrainer):
    model_type = "flux3_klein"
    label = "FLUX.3 [klein]"
    default_name = "flux3_klein_lora"
    batch_one_reason = (
        "FLUX.3 trains at batch_size 1 (every caption keeps its own 80-token text window, "
        "as at inference); use train.gradient_accumulation_steps for a larger effective batch."
    )
    # As for Krea 2: attention, MLP and I/O each adapt their own rate, over a 16-step sign window,
    # which suits batch-1 gradients better than 8.
    polarity_history = 16
    key_commands = False
    schedule = ("uniform", FLUX3_SHIFT)

    # ------------------------------------------------------------------ caching

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        train_config = self.config.get("train", {})
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 512))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()

        vae = load_flux2_vae_encoder(model_config["vae_path"], self.device)
        sizes: dict[tuple[int, int], int] = {}
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                latent = vae.encode(pixels)[0].cpu()
                self.latent_cache[str(source.path)] = latent
                self.samples.append((str(source.path), source.caption))
                size = (pixels.shape[-1], pixels.shape[-2])
                sizes[size] = sizes.get(size, 0) + 1
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        free_memory()
        logger.info(
            "Training sizes (w x h: images): %s", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items()))
        )
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        encoder = load_flux3_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        self._encode_prompts(encoder, prompts)
        del encoder
        free_memory()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info(
            "Pre-cached %d images and %d prompt embeddings (text windows: %s tokens).",
            len(self.samples),
            len(self.text_cache),
            sorted({embeds.shape[0] for embeds in self.text_cache.values()}),
        )

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading FLUX.3 [klein] DiT ===", flush=True)
        model_config = self.config["model"]
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("train", {})
        grad_precision = str(train_config.get("int8_grad_precision", "bf16"))
        self.transformer, load_info = load_flux3_transformer(
            model_config["transformer_path"], self.device, self.dtype, grad_precision=grad_precision
        )
        self.run_metadata["base_load"] = load_info
        self.run_metadata["int8_grad_precision"] = grad_precision
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()

        self.lora_network = Flux3LoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "blocks")),
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self._lora_ready()

    # ------------------------------------------------------------------ training

    def _sample_sigma(self) -> torch.Tensor:
        """Flow-matching sigma for one sample: a uniform or logit-normal draw, then the FLUX.3 shift."""
        train_config = self.config["train"]
        distribution = str(train_config.get("timestep_distribution", "uniform")).lower()
        if distribution in LOGIT_NORMAL_NAMES:
            base = logit_normal(train_config, self.device)
        elif distribution == "uniform":
            base = torch.rand(1, device=self.device)
        else:
            raise ValueError(f"train.timestep_distribution must be uniform or logit_normal, got {distribution!r}")
        return clamped(shifted(base, float(train_config.get("timestep_shift", FLUX3_SHIFT))), train_config)

    def _training_step(self, index: int) -> torch.Tensor:
        # caption dropout trains on the empty prompt in its own 80-token window, exactly the negative prompt CFG uses
        key, caption = self._sample(index)
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        sigma = self._sample_sigma()
        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = (1.0 - s) * x0 + s * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(self.config["train"].get("loss_weight_strength", 1.0))
        return loss * strength if strength != 1.0 else loss


def main() -> None:
    args = training_arguments("Train a FLUX.3 [klein] LoRA in maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    Flux3KleinLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
