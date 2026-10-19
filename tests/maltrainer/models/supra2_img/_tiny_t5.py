from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import T5Config, T5EncoderModel

WORDS = ["a", "red", "cat", "sat", "on", "mat"]
SPECIAL = ["<pad>", "</s>", "<unk>"]


def write_tokenizer(path: Path) -> Path:
    """A word-level tokenizer: ids 0..2 are pad/eos/unk, the words follow."""
    vocab = {token: index for index, token in enumerate(SPECIAL + WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))
    return path


def tiny_config() -> T5Config:
    return T5Config(
        vocab_size=len(SPECIAL + WORDS) + 100,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_heads=2,
        relative_attention_num_buckets=4,
        feed_forward_proj="gated-gelu",
        decoder_start_token_id=0,
    )


def tiny_t5(seed: int = 0) -> T5EncoderModel:
    torch.manual_seed(seed)
    return T5EncoderModel(tiny_config()).eval()


def write_weights(path: Path, model: T5EncoderModel, extra: dict | None = None, drop: tuple = ()) -> Path:
    tensors = {k: v.clone().contiguous() for k, v in model.state_dict().items() if not k.startswith(drop)}
    tensors.update(extra or {})
    save_file(tensors, str(path))
    return path
