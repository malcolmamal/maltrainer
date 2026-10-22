"""The plain (non-quantized) text-encoder path against a real, tiny Qwen3-VL: no fake model class."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLModel

from maltrainer.models.minimax_h3 import text_encoder as te
from tests.maltrainer.models.minimax_h3.h3_text import FakeTokenizer


def _tiny_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": te.MINIMAX_H3_TEXT_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 16,
        },
    )


@pytest.fixture
def plain_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, torch.Tensor]]:
    config = _tiny_config()
    torch.manual_seed(0)
    source = Qwen3VLModel(config).to(torch.bfloat16)
    # The on-disk layout of an upstream text encoder: the text tower under "model.", the vision tower under "visual.".
    state = {
        ("model." + key.removeprefix("language_model.") if key.startswith("language_model.") else key): value
        for key, value in source.state_dict().items()
    }
    path = tmp_path / "te.safetensors"
    save_file(state, str(path))
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(te.Qwen3VLConfig, "from_pretrained", staticmethod(lambda *_a, **_k: _tiny_config()))
    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer)
    return path, state


def test_a_plain_checkpoint_loads_its_weights_into_the_text_encoder(
    plain_checkpoint: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    path, state = plain_checkpoint

    loaded = te.load_minimax_h3_text_encoder(path, device="cpu", dtype=torch.bfloat16)

    parameters = dict(loaded.model.named_parameters())
    assert not any(parameter.is_meta for parameter in parameters.values())
    wanted = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(parameters["language_model.layers.0.self_attn.q_proj.weight"], state[wanted])
    assert torch.equal(parameters["visual.blocks.0.attn.qkv.weight"], state["visual.blocks.0.attn.qkv.weight"])


def test_a_plain_checkpoint_missing_a_text_tower_weight_is_refused(
    plain_checkpoint: tuple[Path, dict[str, torch.Tensor]], tmp_path: Path
) -> None:
    _, state = plain_checkpoint
    del state["model.layers.0.self_attn.q_proj.weight"]
    short = tmp_path / "short.safetensors"
    save_file(state, str(short))

    with pytest.raises(ValueError, match=r"language_model\.layers\.0\.self_attn\.q_proj\.weight"):
        te.load_minimax_h3_text_encoder(short, device="cpu", dtype=torch.bfloat16)
