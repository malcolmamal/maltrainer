"""A tiny Qwen3-VL text model and the loader settings that make it stand in for the 4B encoder."""

from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from maltrainer.models.krea2 import text_encoder as te

TAPS = [1, 2, 3]  # three decoder layers; the deepest tap is the model output
TINY_TEXT = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
}


def tiny_config():
    config = Qwen3VLConfig.from_pretrained(str(te.TOKENIZER_ASSETS)).text_config
    for name, value in TINY_TEXT.items():
        setattr(config, name, value)
    config.num_hidden_layers = TAPS[-1]
    config.use_cache = False
    return config


def tiny_text_model(seed: int = 0) -> Qwen3VLTextModel:
    torch.manual_seed(seed)
    model = Qwen3VLTextModel(tiny_config()).eval()
    model.norm = nn.Identity()
    return model.requires_grad_(False)
