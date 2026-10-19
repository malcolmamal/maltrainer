"""
Qwen3-VL text encoder wrapper for MiniMax H3.
Loads BF16 or Comfy NVFP4 AWQ text encoder checkpoints and extracts layer-50 hidden states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3VLConfig, Qwen3VLModel

from maltrainer.models.minimax_h3.comfy_quant import has_comfy_quantized_layers, load_comfy_quantized_state_dict

logger = logging.getLogger(__name__)

QWEN3VL_BOS_TOKEN_ID = 151643

# Qwen3-VL tokenizer + config metadata, vendored in-tree. maltrainer must not depend on a
# sibling trainer checkout being present: see assets/text_encoder/ASSET_PROVENANCE.md for
# the upstream source and pinned revision. model.tokenizer_path overrides this.
BUNDLED_TOKENIZER_ASSETS = Path(__file__).resolve().parent / "assets" / "text_encoder"


def _resolve_tokenizer_assets(explicit: str | Path | None) -> Path:
    """Return the directory holding the Qwen3-VL tokenizer assets."""
    candidate = Path(explicit) if explicit is not None else BUNDLED_TOKENIZER_ASSETS
    if not (candidate / "tokenizer.json").exists():
        source = "model.tokenizer_path" if explicit is not None else "the bundled assets directory"
        raise FileNotFoundError(
            f"No tokenizer.json in {candidate} (from {source}). The H3 Qwen3-VL tokenizer "
            "metadata ships in maltrainer/models/minimax_h3/assets/text_encoder; if it is "
            "missing, restore it from git or point model.tokenizer_path at a copy."
        )
    return candidate


# H3 conditions on the unnormalized hidden state after Qwen3-VL layer 50; the upstream
# config declares the model's full depth.
MINIMAX_H3_TEXT_LAYERS = 50


def _describe_text_config(text_config) -> str:
    """Compact summary of the resolved text tower, for a startup sanity check.

    Read defensively: transformers has reshuffled where RoPE settings live between
    versions (rope_theta/rope_scaling vs. a merged rope_parameters dict), and a log line
    must never be the thing that fails a training run.
    """
    fields = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim"]
    parts = [f"{name}={getattr(text_config, name, '?')}" for name in fields]
    rope = getattr(text_config, "rope_parameters", None)
    if rope is None:
        theta = getattr(text_config, "rope_theta", "?")
        rope = {"rope_theta": theta, "rope_scaling": getattr(text_config, "rope_scaling", None)}
    parts.append(f"rope={rope}")
    return " ".join(parts)


def _text_encoder_key(source_prefix: str) -> str:
    if source_prefix.startswith("visual"):
        return source_prefix
    if source_prefix == "model":
        return "language_model"
    if source_prefix.startswith("model."):
        return f"language_model.{source_prefix.removeprefix('model.')}"
    return source_prefix


class MiniMaxH3TextEncoder(nn.Module):
    """Wrapper for Qwen3-VL text encoder extracting layer-50 hidden states."""

    def __init__(self, text_encoder_model: Qwen3VLModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = text_encoder_model
        self.tokenizer = tokenizer
        self._prompt_cache: dict[str, torch.Tensor] = {}
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: list[str],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 1024,
    ) -> torch.Tensor:
        """Encode prompts to unnormalized layer-50 text hidden states [B, S, 5120].

        H3's text presentation is raw prompt text: no chat template and no special
        tokens. ComfyUI tokenizes exactly this way, so a chat-templated training prompt
        prepends ~30 tokens that never exist at inference -- which also shifts the packed
        sequence's video RoPE origin, since that origin is text_len.
        """
        cache_key = f"{dtype}_" + "||".join(prompts)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key].to(device)

        self.tokenizer.padding_side = "right"
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if encoded.input_ids.shape[1] == 0:
            # H3 represents an otherwise empty prompt by Qwen3-VL's BOS token.
            input_ids = torch.full((len(prompts), 1), QWEN3VL_BOS_TOKEN_ID, device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            empty_rows = attention_mask.sum(dim=1) == 0
            if empty_rows.any():
                input_ids[empty_rows, 0] = QWEN3VL_BOS_TOKEN_ID
                attention_mask[empty_rows, 0] = 1

        language_model = getattr(self.model, "language_model", self.model)
        outputs = language_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        # Drop right padding so the cached text length equals ComfyUI's at inference.
        lengths = attention_mask.sum(dim=1)
        if int(lengths.min()) != int(lengths.max()):
            raise ValueError(
                "encode_prompts expects prompts of equal token length in one call; "
                "encode captions one at a time so padding never enters the cache"
            )
        hidden_states = hidden_states[:, : int(lengths[0])]

        res = hidden_states.to(dtype=dtype)
        self._prompt_cache[cache_key] = res
        return res


def load_minimax_h3_text_encoder(
    text_encoder_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3TextEncoder:
    """Load Qwen3-VL text encoder (BF16 or NVFP4/AWQ) and tokenizer."""
    text_encoder_path = Path(text_encoder_path)
    device = torch.device(device)

    # Qwen3-VL tokenizer assets. Substituting a different tokenizer would change token ids
    # outright, so a miss raises instead of silently falling back to something plausible.
    asset_dir = _resolve_tokenizer_assets(tokenizer_path)
    logger.info("Using MiniMax H3 tokenizer assets from %s", asset_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = QWEN3VL_BOS_TOKEN_ID

    # Take the whole config from the vendored upstream config.json rather than
    # hand-transcribing it: from_pretrained also migrates the JSON to whatever schema the
    # installed transformers expects, which hand-written kwargs do not. The only H3-specific
    # change is truncating the text tower to the layer it conditions on.
    config = Qwen3VLConfig.from_pretrained(str(asset_dir))
    text_config: Any = config.text_config
    text_config.num_hidden_layers = MINIMAX_H3_TEXT_LAYERS
    text_config.use_cache = False
    logger.info("H3 text tower: %s", _describe_text_config(text_config))

    with init_empty_weights(include_buffers=True):
        model: Any = Qwen3VLModel(config)  # PreTrainedModel.to is typed as an unbound _Wrapped
        model.language_model.norm = nn.Identity()

    if has_comfy_quantized_layers(text_encoder_path):
        logger.info("Loading Comfy NVFP4/AWQ quantized text encoder from %s", text_encoder_path)
        load_comfy_quantized_state_dict(model, text_encoder_path, key_map=_text_encoder_key, output_dtype=dtype)
    else:
        logger.info("Loading text encoder safetensors from %s", text_encoder_path)
        from safetensors.torch import load_file

        sd = load_file(str(text_encoder_path))
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith(("visual.", "model.")):
                prefix, _, suffix = k.rpartition(".")
                target_key = f"{_text_encoder_key(prefix)}.{suffix}"
                mapped_sd[target_key] = v
        model.load_state_dict(mapped_sd, strict=False)

    vision_config: Any = config.vision_config
    vision_head_dim = vision_config.hidden_size // vision_config.num_heads
    model.visual.rotary_pos_emb = type(model.visual.rotary_pos_emb)(vision_head_dim // 2).to(device)
    model.language_model.rotary_emb = type(model.language_model.rotary_emb)(text_config, device=device)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return MiniMaxH3TextEncoder(model, tokenizer)
