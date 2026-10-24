"""The trainable model families the backend offers, each with its script, template and defaults."""

from __future__ import annotations

from fastapi import HTTPException

from backend.registry_fibo2 import FIBO2
from backend.specs import ModelSpec, OptimizerSpec

MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="zimage",
            label="Z-Image (Base)",
            script="train_zimage_lora.py",
            template="templates/zimage_base_template.yaml",
            optimizers=(
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4),
                OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
            ),
            default_optimizer="adamw8bit",
            default_duration_mode="epochs",
            default_epochs=100,
            default_max_steps=2000,
            native_max_steps=False,
        ),
        ModelSpec(
            id="qwen_image_2_1",
            label="Qwen-Image 2.1",
            script="train_qwen_image_2_1_lora.py",
            template="templates/qwen_image_2_1_template.yaml",
            optimizers=(
                OptimizerSpec(
                    "automagic3",
                    "Automagic3 (recommended)",
                    1e-6,
                    "self-tuning; the learning rate is only its starting point",
                ),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "keep checkpoints 1250-1750"),
                OptimizerSpec("prodigy", "Prodigy", 1.0, "not recommended on Qwen: overcooks past ~1500 steps"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"),
                ("vae_path", "vae", "qwen_image_2.1_vae_bf16.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),
            ),
            sec_per_step=0.62,
            setup_seconds=20.0,
            per_run_output=True,
            target_epochs=70,
            min_auto_steps=1000,
            max_auto_steps=2500,
            epoch_bounds=(20, 120),
            max_benchmarked_steps=2500,
            square_only=True,
            resolution_note=(
                "Training resolution is square 512×512 for now. More resolutions and "
                "aspect-ratio bucketing are planned."
            ),
            output_prefix="qwen21_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="krea2",
            label="Krea 2",
            script="train_krea2_lora.py",
            template="templates/krea2_template.yaml",
            optimizers=(
                OptimizerSpec(
                    "automagic3",
                    "Automagic3 (recommended)",
                    1e-6,
                    "self-tuning; per module family, 16-step sign window",
                ),
                OptimizerSpec(
                    "prodigy",
                    "Prodigy (OneTrainer recipe)",
                    1.0,
                    "rank 16 / alpha 1, 200 warmup steps, unshifted logit-normal",
                ),
                OptimizerSpec(
                    "adamw8bit", "AdamW 8-bit (Fizgig Standard)", 1e-4, "rank 32, every Linear, flat 1e-4, EMA 0.98"
                ),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "krea2_raw_fp8_scaled.safetensors"),
                ("vae_path", "vae", "qwen_image_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            optimizer_templates=(
                ("prodigy", "templates/krea2_prodigy_template.yaml"),
                ("adamw8bit", "templates/krea2_adamw_template.yaml"),
            ),
            output_prefix="krea2_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="flux3_klein",
            label="FLUX.3 [klein] (Base)",
            script="train_flux3_klein_lora.py",
            template="templates/flux3_klein_template.yaml",
            optimizers=(
                OptimizerSpec(
                    "automagic3",
                    "Automagic3 (recommended)",
                    1e-6,
                    "self-tuning; per module family, 16-step sign window",
                ),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "the diffusers FLUX.3 example's rate"),
                OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "flux3_image_klein_int8_convrot.safetensors"),
                ("vae_path", "vae", "flux2_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            output_prefix="flux3_klein_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="moss_image",
            label="MOSS-Image (Base)",
            script="train_moss_image_lora.py",
            template="templates/moss_image_automagic_template.yaml",
            optimizers=(
                OptimizerSpec(
                    "automagic3", "Automagic3 (recommended)", 1e-6, "self-tuning; per module family, 8-step sign window"
                ),
                OptimizerSpec(
                    "prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier; 100 warmup steps, no EMA"
                ),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "flat rate, weight decay 0.01"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "moss_image_base_bf16.safetensors"),
                ("vae_path", "vae", "flux2_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3.5_9b_int8_convrot.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            optimizer_templates=(
                ("prodigy", "templates/moss_image_prodigy_template.yaml"),
                ("adamw8bit", "templates/moss_image_adamw_template.yaml"),
            ),
            output_prefix="moss_image_",
            caption_from_trigger=True,
        ),
        FIBO2,
    )
}


def get_model_spec(model_id: str) -> ModelSpec:
    spec = MODELS.get(model_id)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}")
    return spec
