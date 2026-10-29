"""LTX-2.5 (dev) in the backend's model list."""

from __future__ import annotations

from backend.specs import ModelSpec, OptimizerSpec

LTX25 = ModelSpec(
    id="ltx25",
    label="LTX-2.5 (Dev, video + audio)",
    script="train_ltx25_lora.py",
    template="templates/ltx25_template.yaml",
    optimizers=(
        OptimizerSpec(
            "automagic3",
            "Automagic3 (recommended)",
            1e-6,
            "self-tuning; per module family, 16-step sign window",
        ),
        OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "the official LTX-2 LoRA trainer's rate"),
        OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
    ),
    default_optimizer="automagic3",
    default_duration_mode="steps",
    default_epochs=70,
    default_max_steps=2000,
    native_max_steps=True,
    model_files=(
        ("transformer_path", "diffusion_models", "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"),
        ("video_vae_path", "vae", "ltx-2.5-video-vae-conv-bf16.safetensors"),
        ("audio_vae_path", "vae", "ltx-2.5-audio-vae-bf16.safetensors"),
        ("text_encoder_path", "text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ),
    per_run_output=True,
    resolution_note="Aspect-ratio buckets around 512×512 on a 32 px grid; clips of up to 49 frames at 24 fps.",
    output_prefix="ltx25_",
    caption_from_trigger=True,
)
