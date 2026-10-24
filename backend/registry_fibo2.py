"""Fibo 2.0 (Dev) in the backend's model list."""

from __future__ import annotations

from backend.specs import ModelSpec, OptimizerSpec

FIBO2 = ModelSpec(
    id="fibo2",
    label="Fibo 2.0 (Dev)",
    script="train_fibo2_lora.py",
    template="templates/fibo2_template.yaml",
    optimizers=(
        OptimizerSpec(
            "automagic3",
            "Automagic3 (recommended)",
            1e-6,
            "self-tuning; per module family, 16-step sign window",
        ),
        OptimizerSpec(
            "adamw",
            "AdamW (BRIA's recipe)",
            1e-4,
            "BRIA's own settings: rank 64 on every Linear, 8 accumulated samples a step; about 1000 steps",
        ),
        OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "the Automagic3 template's settings with AdamW 8-bit"),
        OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
    ),
    default_optimizer="automagic3",
    optimizer_templates=(("adamw", "templates/fibo2_bria_template.yaml"),),
    default_duration_mode="steps",
    default_epochs=70,
    default_max_steps=1500,
    native_max_steps=True,
    model_files=(
        ("transformer_path", "diffusion_models", "fibo2_dev_bf16.safetensors"),
        ("vae_path", "vae", "flux2_vae.safetensors"),
        ("text_encoder_path", "text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
    ),
    per_run_output=True,
    resolution_note="Aspect-ratio buckets around 1024×1024 on a 16 px grid.",
    output_prefix="fibo2_",
    caption_from_trigger=True,
)
