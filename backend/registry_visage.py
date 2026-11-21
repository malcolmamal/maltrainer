"""Visage-Image in the backend's model list."""

from __future__ import annotations

from backend.specs import ModelSpec, OptimizerSpec

VISAGE = ModelSpec(
    id="visage",
    label="Visage-Image (256 px faces)",
    script="train_visage_lora.py",
    template="templates/visage_template.yaml",
    optimizers=(
        OptimizerSpec(
            "automagic3",
            "Automagic3 (recommended)",
            1e-6,
            "self-tuning; per module family, 16-step sign window",
        ),
        OptimizerSpec("adamw", "AdamW", 1e-4, "a flat rate, weight decay 0.01; the usual LoRA starting point"),
    ),
    default_optimizer="automagic3",
    optimizer_templates=(("adamw", "templates/visage_adamw_template.yaml"),),
    default_duration_mode="steps",
    default_epochs=70,
    default_max_steps=1500,
    native_max_steps=True,
    model_files=(
        ("transformer_path", "diffusion_models", "visage-image.safetensors"),
        ("vae_path", "vae", "visage_vae.safetensors"),
        ("text_encoder_path", "text_encoders", "qwen3_0.6b.safetensors"),
    ),
    per_run_output=True,
    resolution_note="256×256 only: each image is centre-cropped to a square, so crop faces yourself where it matters.",
    output_prefix="visage_",
    caption_from_trigger=True,
)
