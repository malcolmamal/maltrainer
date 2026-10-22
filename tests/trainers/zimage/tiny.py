"""Tiny but real Z-Image components: diffusers' transformer and VAE and transformers' Qwen3 encoder at toy size."""

from __future__ import annotations

from typing import Any, cast

import torch
from diffusers import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

LATENT_CHANNELS = 4
HIDDEN = 16
WORDS = ["[UNK]", "[PAD]", "<|user|>", "<|assistant|>", "think", "a", "photo", "of", "person", "zed"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|user|> {{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %} <|assistant|>{% if enable_thinking %} think{% endif %}{% endif %}"
)
TRANSFORMER: dict[str, Any] = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": LATENT_CHANNELS,
    "dim": 16,
    "n_layers": 1,
    "n_refiner_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "cap_feat_dim": HIDDEN,
    "axes_dims": [4, 2, 2],
    "axes_lens": [64, 32, 32],
}


def build_vae(seed: int = 1) -> Any:
    """A 2x-downsampling VAE whose posterior has (almost) no variance, so ``sample()`` is its mean.

    Typed ``Any``: diffusers' hook decorators hide ``encode`` and ``quant_conv`` from mypy.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        vae: Any = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(8, 8),
            latent_channels=LATENT_CHANNELS,
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=16,
        )
        with torch.no_grad():
            vae.quant_conv.weight[LATENT_CHANNELS:] = 0
            vae.quant_conv.bias[LATENT_CHANNELS:] = -30
    return vae


def build_transformer(seed: int = 2) -> ZImageTransformer2DModel:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ZImageTransformer2DModel(**TRANSFORMER)


def build_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def build_text_encoder(seed: int = 3) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=len(WORDS),
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return Qwen3ForCausalLM(config)


def embedding_table(model: Any) -> torch.Tensor:
    """The input embedding weights of a transformers model."""
    return cast(torch.Tensor, model.get_input_embeddings().weight)
