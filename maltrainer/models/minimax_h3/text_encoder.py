"""
Qwen3-VL text encoder wrapper for MiniMax H3.
Loads BF16 or Comfy NVFP4 AWQ text encoder checkpoints and extracts layer-50 hidden states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
from accelerate import init_empty_weights

from maltrainer.models.minimax_h3.comfy_quant import has_comfy_quantized_layers, load_comfy_quantized_state_dict

logger = logging.getLogger(__name__)

QWEN3VL_BOS_TOKEN_ID = 151643

# The Qwen3-VL tokenizer + config assets for H3's text encoder ship inside the
# musubi-tuner MiniMax forks. Checkouts get arranged differently per machine
# (.../trainers/<fork>, .../trainers/akane/<fork>, .../sandbox/trainers/<fork>), so rather
# than enumerate layouts we glob a couple of levels up from this repo for the known
# asset subpath. model.tokenizer_path in the training config always wins.
_ASSET_SUBPATH = Path("src") / "musubi_tuner" / "minimax_h3" / "assets" / "text_encoder"
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../maltrainer
_ASSET_GLOBS = (
    str(Path("*") / _ASSET_SUBPATH),           # <root>/<fork>/src/...
    str(Path("*") / "*" / _ASSET_SUBPATH),     # <root>/<group>/<fork>/src/...
)


def _resolve_tokenizer_assets(explicit: Union[str, Path, None]) -> Path | None:
    """Return a directory containing the Qwen3-VL tokenizer assets, or None."""
    if explicit is not None:
        candidate = Path(explicit)
        if (candidate / "tokenizer.json").exists():
            return candidate
        raise FileNotFoundError(f"model.tokenizer_path has no tokenizer.json: {candidate}")

    search_roots = [_REPO_ROOT.parent, _REPO_ROOT.parent.parent]
    for root in search_roots:
        for pattern in _ASSET_GLOBS:
            # sorted() keeps the pick deterministic when several forks are checked out
            for candidate in sorted(root.glob(pattern)):
                if (candidate / "tokenizer.json").exists():
                    return candidate
    return None


def _minimax_h3_text_config():
    """Qwen3-VL-32B text config truncated to the 50 layers H3 conditions on.

    Pinned explicitly rather than read from a checkpoint sibling: these values decide the
    attention head layout and the mRoPE schedule, and a config fallback that silently
    disagreed with the checkpoint would yield valid-looking but wrong text embeddings.
    Mirrors musubi-tuner's minimax_h3/text_encoder.py::_text_config.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    return Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=50,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000,
        rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )


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

    def __init__(self, text_encoder_model: Qwen3VLModel, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = text_encoder_model
        self.tokenizer = tokenizer
        self._prompt_cache: dict[str, torch.Tensor] = {}
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: List[str],
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
    text_encoder_path: Union[str, Path],
    tokenizer_path: Union[str, Path, None] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3TextEncoder:
    """Load Qwen3-VL text encoder (BF16 or NVFP4/AWQ) and tokenizer."""
    text_encoder_path = Path(text_encoder_path)
    device = torch.device(device)

    # Qwen3-VL tokenizer assets. Falling back to a Qwen2-VL tokenizer would change token
    # ids outright, so a miss is fatal rather than silently substituted.
    asset_dir = _resolve_tokenizer_assets(tokenizer_path)
    if asset_dir is None:
        searched = ", ".join(str(root) for root in (_REPO_ROOT.parent, _REPO_ROOT.parent.parent))
        raise FileNotFoundError(
            "Could not find the Qwen3-VL tokenizer assets for MiniMax H3. Looked for "
            f"*/{_ASSET_SUBPATH} and */*/{_ASSET_SUBPATH} under: {searched}. "
            "These assets ship with the musubi-tuner MiniMax forks (e.g. "
            "musubi-tuner-akane-minimax). Set model.tokenizer_path in the training config to "
            "the directory containing tokenizer.json."
        )
    logger.info("Using MiniMax H3 tokenizer assets from %s", asset_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = QWEN3VL_BOS_TOKEN_ID

    # Build Qwen3-VL config for the 50-layer 32B text tower, pinning the text side.
    try:
        config = Qwen3VLConfig.from_pretrained(str(asset_dir))
    except Exception as error:
        logger.warning("Could not read Qwen3-VL config from %s (%s); using defaults.", asset_dir, error)
        config = Qwen3VLConfig()

    config.text_config = _minimax_h3_text_config()
    logger.info(
        "H3 text tower: layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d rope_theta=%s",
        config.text_config.num_hidden_layers,
        config.text_config.hidden_size,
        config.text_config.num_attention_heads,
        config.text_config.num_key_value_heads,
        config.text_config.head_dim,
        config.text_config.rope_theta,
    )

    with init_empty_weights(include_buffers=True):
        model = Qwen3VLModel(config)
        model.language_model.norm = nn.Identity()

    if has_comfy_quantized_layers(text_encoder_path):
        logger.info("Loading Comfy NVFP4/AWQ quantized text encoder from %s", text_encoder_path)
        load_comfy_quantized_state_dict(
            model,
            text_encoder_path,
            key_map=_text_encoder_key,
            output_dtype=dtype,
        )
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

    vision_config = config.vision_config
    vision_head_dim = vision_config.hidden_size // vision_config.num_heads
    model.visual.rotary_pos_emb = type(model.visual.rotary_pos_emb)(vision_head_dim // 2).to(device)
    model.language_model.rotary_emb = type(model.language_model.rotary_emb)(
        config.text_config,
        device=device,
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return MiniMaxH3TextEncoder(model, tokenizer)
