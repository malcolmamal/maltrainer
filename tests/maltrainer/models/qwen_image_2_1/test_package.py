from __future__ import annotations

import maltrainer.models.qwen_image_2_1 as package
from maltrainer.models.qwen_image_2_1 import lora, model, text_encoder, vae


def test_the_package_re_exports_the_public_model_vae_text_encoder_and_lora_names() -> None:
    assert package.QwenImage21Transformer is model.QwenImage21Transformer
    assert package.calculate_shift is model.calculate_shift
    assert package.apply_flow_shift is model.apply_flow_shift
    assert package.build_rope_frequencies is model.build_rope_frequencies
    assert package.QwenImage21VAE is vae.QwenImage21VAE
    assert package.load_qwen_image_2_1_vae is vae.load_qwen_image_2_1_vae
    assert package.Qwen3VLTextEncoder is text_encoder.Qwen3VLTextEncoder
    assert package.load_qwen_image_2_1_text_encoder is text_encoder.load_qwen_image_2_1_text_encoder
    assert package.QwenImage21LoRALayer is lora.QwenImage21LoRALayer
    assert package.QwenImage21LoRANetwork is lora.QwenImage21LoRANetwork


def test_every_name_in_all_exists() -> None:
    assert all(hasattr(package, name) for name in package.__all__)
    assert len(package.__all__) == len(set(package.__all__)) == 10
