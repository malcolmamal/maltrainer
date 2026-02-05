#!/usr/bin/env python3
"""
Z-Image Textual Inversion Training Script
Trains textual inversion embeddings for Z-Image (base) models with YAML configuration support.
"""

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM

from maltrainer.models.embedding import TextualInversionEmbedding
from maltrainer.models.scheduler import get_flow_match_scheduler
from maltrainer.data.dataset import ZImageDataset
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ZImageEmbeddingTrainer:
    """Trainer for Z-Image Textual Inversion embeddings."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )
        
        self.is_main_process = self.accelerator.is_main_process

        # Log device info
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info(f"Training on device: {self.device} ({device_name})")
            if self.device.type == "cuda":
                logger.info(f"Total VRAM: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB")
        
    def load_model(self):
        """Load Z-Image model components."""
        logger.info("Loading Z-Image model...")
        
        model_path = self.config["model"]["name_or_path"]
        
        # Load transformer
        self.transformer = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        self.transformer.enable_gradient_checkpointing()
        
        # Load text encoder
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE always float32
        )
        
        # Set requires_grad
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        # Enable gradients for embedding layer
        self.text_encoder.get_input_embeddings().requires_grad_(True)
        
        # Move to device
        self.transformer.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        logger.info("Model loaded successfully")
        
    def create_embedding(self):
        """Create textual inversion embedding."""
        logger.info("Creating textual inversion embedding...")
        
        embedding_config = self.config.get("embedding", {})
        
        self.embedding = TextualInversionEmbedding(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            trigger_token=embedding_config.get("trigger", "sks"),
            num_vectors=embedding_config.get("num_vectors", 1),
            init_text=embedding_config.get("init_text", "a photo"),
        )
        
        # Backup original embeddings for regularization
        self.original_embeddings = self.text_encoder.get_input_embeddings().weight.data.clone()
        
        logger.info(f"Embedding created with trigger '{embedding_config.get('trigger', 'sks')}'")
        
    def create_dataset(self):
        """Create training dataset."""
        logger.info("Creating dataset...")
        
        dataset_config = self.config["dataset"]
        embedding_config = self.config.get("embedding", {})
        
        self.train_dataset = ZImageDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            center_crop=dataset_config.get("center_crop", True),
            random_flip=dataset_config.get("random_flip", False),
            trigger_token=embedding_config.get("trigger", "sks"),
            repeats=dataset_config.get("repeats", 1),
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"].get("batch_size", 1),
            shuffle=True,
            num_workers=dataset_config.get("num_workers", 0),
        )
        
        logger.info(f"Dataset created with {len(self.train_dataset)} samples")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        train_config = self.config["train"]
        
        def _to_float(value, default: float) -> float:
            if value is None:
                return float(default)
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))

        lr = _to_float(train_config.get("learning_rate", 5e-4), 5e-4)
        beta1 = _to_float(train_config.get("adam_beta1", 0.9), 0.9)
        beta2 = _to_float(train_config.get("adam_beta2", 0.999), 0.999)
        weight_decay = _to_float(train_config.get("weight_decay", 0.01), 0.01)
        eps = _to_float(train_config.get("adam_epsilon", 1e-8), 1e-8)

        # Optimize the entire embedding layer
        # Note: We rely on restore_original_embeddings to keep non-trained tokens frozen
        # Passing slices (non-leaf tensors) caused warnings and prevented updates
        params_to_optimize = [self.text_encoder.get_input_embeddings().weight]

        self.optimizer = AdamW(
            params_to_optimize,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=eps,
        )
        
        self.noise_scheduler = get_flow_match_scheduler()
        
    def restore_original_embeddings(self):
        """Restore original embeddings except for the learned token."""
        with torch.no_grad():
            token_ids = self.embedding.token_ids
            index_no_updates = torch.ones(len(self.tokenizer), dtype=torch.bool)
            
            for token_id in token_ids:
                index_no_updates[token_id] = False
            
            # Restore all embeddings except the learned ones
            self.text_encoder.get_input_embeddings().weight[index_no_updates] = \
                self.original_embeddings[index_no_updates]
        
    def train(self):
        """Main training loop."""
        train_config = self.config["train"]
        num_epochs = train_config.get("num_epochs", 10)
        save_every = train_config.get("save_every", 500)
        
        # Log device and model info
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            print(f"--- Training Configuration ---")
            print(f"Device: {device_name}")
            print(f"Dtype: {self.dtype}")
            print(f"Transformer: {sum(p.numel() for p in self.transformer.parameters())/1e9:.2f}B params")
            print(f"Text Encoder: {sum(p.numel() for p in self.text_encoder.parameters())/1e9:.2f}B params")
            print(f"-------------------------------")

        # Prepare for training
        self.embedding, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.embedding, self.optimizer, self.train_dataloader
        )
        
        global_step = 0
        
        logger.info("Starting training...")
        
        # Calculate total steps and setup progress bar
        num_update_steps_per_epoch = len(self.train_dataloader)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        
        progress_bar = tqdm(
            range(max_train_steps),
            desc="Steps",
            disable=not self.is_main_process,
        )
        
        for epoch in range(num_epochs):
            self.embedding.train()
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.embedding):
                    # Get pixel values and captions from batch
                    pixel_values = batch["pixel_values"].to(self.device)
                    captions = batch["caption"]
                    
                    # Encode with VAE
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        # Z-Image specific scaling/shifting logic
                        zimage_shift = 0.1159
                        zimage_scale = 0.3611
                        latents = (latents - zimage_shift) * zimage_scale
                        latents = latents.to(dtype=self.dtype)
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        [
                            self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": c}],
                                tokenize=False,
                                add_generation_prompt=True,
                                **({"enable_thinking": True} if "enable_thinking" in inspect.signature(self.tokenizer.apply_chat_template).parameters or "kwargs" in inspect.signature(self.tokenizer.apply_chat_template).parameters else {})
                            )
                            if hasattr(self.tokenizer, "apply_chat_template")
                            else c
                            for c in captions
                        ],
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    # Encode text
                    # We need gradients to flow through text encoder input embeddings
                    base_text_encoder = getattr(self.text_encoder, "model", self.text_encoder)
                    te_out = base_text_encoder(
                        tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        output_hidden_states=True,
                    )
                    text_embeds = te_out.hidden_states[-2]
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    
                    # Sample timestep
                    bsz = latents.shape[0]
                    timesteps = torch.rand(bsz, device=self.device)
                    
                    # Add noise to latents (flow matching formulation)
                    noisy_latents = (1 - timesteps[:, None, None, None]) * latents + timesteps[:, None, None, None] * noise
                    noisy_latents = noisy_latents.to(dtype=self.dtype)
                    
                    # Predict velocity
                    transformer_kwargs = {}
                    forward_params = inspect.signature(self.transformer.forward).parameters
                    if "cap_mask" in forward_params:
                        transformer_kwargs["cap_mask"] = tokens.attention_mask.bool()
                    
                    output = self.transformer(
                        noisy_latents.unsqueeze(2),
                        1.0 - timesteps,
                        text_embeds,
                        **transformer_kwargs,
                    )
                    
                    # Handle different return types (BaseOutput, tuple, list)
                    if hasattr(output, "sample"):
                        model_pred = output.sample
                    elif isinstance(output, (list, tuple)):
                        model_pred = output[0]
                    else:
                        model_pred = output
                        
                    # If it's still a list (some models return [tensor]), take the first element
                    if isinstance(model_pred, (list, tuple)):
                        model_pred = model_pred[0]
                    
                    # Squeeze the frame dimension if it exists
                    if len(model_pred.shape) == 5:
                        model_pred = model_pred.squeeze(2)

                    # Fix potential channel/batch swap to avoid broadcasting
                    if (
                        model_pred.ndim == 4
                        and model_pred.shape[0] == latents.shape[1]
                        and model_pred.shape[1] == latents.shape[0]
                        and model_pred.shape[2:] == latents.shape[2:]
                    ):
                        model_pred = model_pred.permute(1, 0, 2, 3).contiguous()
                    
                    # Compute loss
                    # Z-Image / Musubi uses target = latents - noise
                    target = latents - noise
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Backpropagate
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        # Clip gradients on the actual weight tensor we are optimizing
                        self.accelerator.clip_grad_norm_(
                            [self.text_encoder.get_input_embeddings().weight],
                            train_config.get("max_grad_norm", 1.0)
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Restore original embeddings (regularization)
                    self.restore_original_embeddings()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)
                
                # Update progress bar logs
                if global_step % 10 == 0:  # Update logs less frequently
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch+1}/{num_epochs}"})
        
        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training completed!")
        
    def save_checkpoint(self, step):
        """Save embedding checkpoint."""
        if not self.is_main_process:
            return
        
        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / f"zimage_embedding_step_{step}.safetensors"
        
        # Get embedding vectors
        state_dict = self.embedding.get_state_dict()
        
        # Add metadata
        embedding_config = self.config.get("embedding", {})
        metadata = {
            "step": str(step),
            "trigger": embedding_config.get("trigger", "sks"),
            "num_vectors": str(embedding_config.get("num_vectors", 1)),
            "init_text": embedding_config.get("init_text", "a photo"),
        }
        
        save_file(state_dict, str(save_path), metadata=metadata)
        logger.info(f"Saved embedding to {save_path}")
        
    def run(self):
        """Run the complete training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            
            self.load_model()
            self.create_embedding()
            self.create_dataset()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise e


def main():
    parser = argparse.ArgumentParser(description="Train Z-Image Textual Inversion Embedding")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.get("logging", {}))
    
    # Save config to output directory
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    
    # Create trainer and run
    trainer = ZImageEmbeddingTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
