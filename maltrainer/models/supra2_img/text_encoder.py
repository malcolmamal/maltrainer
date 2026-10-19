"""Frozen Flan-T5-Base encoder for Supra2-IMG. Context length is 128."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel, T5TokenizerFast

from maltrainer.models.supra2_img.model import MAX_CTX_LEN


class FlanT5Encoder:
    def __init__(self, model: T5EncoderModel, tokenizer: T5TokenizerFast, ctx_len: int = MAX_CTX_LEN):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len

    @torch.no_grad()
    def encode(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.ctx_len,
            return_tensors="pt",
        ).to(device)
        use_amp = device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            hidden = self.model(**tokens).last_hidden_state
        return hidden.float(), tokens["attention_mask"].float()


def load_flan_t5(
    weights_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    device: torch.device,
    ctx_len: int = MAX_CTX_LEN,
) -> FlanT5Encoder:
    tokenizer = T5TokenizerFast(tokenizer_file=str(tokenizer_path), extra_ids=100)
    model = T5EncoderModel(T5Config.from_json_file(str(config_path)))
    state = load_file(str(weights_path), device="cpu")
    incompat = model.load_state_dict(state, strict=False)
    missing = [key for key in incompat.missing_keys if "embed_tokens" not in key]
    unexpected = [
        key for key in incompat.unexpected_keys if not key.startswith("decoder.") and key != "lm_head.weight"
    ]
    if missing or unexpected:
        raise RuntimeError(f"Flan-T5 load mismatch missing={missing} unexpected={unexpected}")
    model.to(device=device)
    model.requires_grad_(False)
    model.eval()
    return FlanT5Encoder(model, tokenizer, ctx_len=ctx_len)
