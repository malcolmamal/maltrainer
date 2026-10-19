from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from maltrainer.models.flux3 import text_encoder as te
from maltrainer.models.flux3.text_encoder import load_flux3_text_encoder
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding
from tests.maltrainer.models.flux3.checkpoints import marker, write

CPU = torch.device("cpu")
TINY_TEXT = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
}
PREFIX = "model.language_model."


@pytest.fixture(autouse=True)
def tiny_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(te, "QWEN3VL_4B_TEXT", TINY_TEXT)
    monkeypatch.setattr(te, "TAP_LAYERS", (1, 2, 3))


def _state(seed: int = 0) -> dict[str, torch.Tensor]:
    config = Qwen3VLConfig.from_pretrained(str(te.TOKENIZER_ASSETS)).text_config
    for name, value in TINY_TEXT.items():
        setattr(config, name, value)
    config.num_hidden_layers = 3
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in Qwen3VLTextModel(config).state_dict().items()}


def _int8_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    state = {k: v.to(torch.bfloat16) for k, v in _state().items()}
    for key in [k for k in state if k.endswith(("_proj.weight", "embed_tokens.weight"))]:
        base = key[: -len(".weight")]
        state[key] = torch.randint(-50, 50, state[key].shape, generator=generator, dtype=torch.int8)
        state[f"{base}.weight_scale"] = torch.full((state[key].shape[0],), 0.02)
        state[f"{base}.comfy_quant"] = marker(format="int8_tensorwise", convrot=True, convrot_groupsize=16)
    return state


def _file(tmp_path: Path, state: dict[str, torch.Tensor], prefix: str = PREFIX) -> Path:
    return write(tmp_path / "qwen.safetensors", state, prefix=prefix)


@pytest.mark.parametrize(
    "prefix", ["", PREFIX, "language_model.model.", "text_encoders.qwen3vl_4b.transformer.", "model."]
)
def test_the_text_model_loads_whatever_prefix_the_file_uses(tmp_path: Path, prefix: str) -> None:
    state = _state()

    encoder = load_flux3_text_encoder(_file(tmp_path, state, prefix), CPU, dtype=torch.float32)

    model = encoder.model
    assert len(model.layers) == 3
    assert torch.equal(model.layers[1].self_attn.q_proj.weight, state["layers.1.self_attn.q_proj.weight"])
    assert isinstance(model.norm, nn.Identity)


def test_vision_head_final_norm_and_deeper_layers_in_the_file_are_ignored(tmp_path: Path) -> None:
    state = _state()
    state["visual.blocks.0.weight"] = torch.zeros(1)
    state["lm_head.weight"] = torch.zeros(1)
    state["layers.3.self_attn.q_proj.weight"] = torch.zeros(1)

    encoder = load_flux3_text_encoder(_file(tmp_path, state), CPU)

    assert len(encoder.model.layers) == 3


def test_the_loaded_encoder_is_frozen_casts_to_dtype_and_encodes_a_prompt(tmp_path: Path) -> None:
    encoder = load_flux3_text_encoder(_file(tmp_path, _state()), CPU, dtype=torch.float32)

    out = encoder.encode("a cat", CPU, dtype=torch.float32)

    assert out.shape == (80, 3 * 16)
    assert torch.isfinite(out).all()
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.parameters())
    assert encoder.model.rotary_emb.inv_freq.device.type == "cpu"
    assert {p.dtype for p in encoder.model.parameters()} == {torch.float32}


def test_a_tokenizer_path_overrides_the_vendored_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", lambda path, **_: seen.append(path) or "tok")

    encoder = load_flux3_text_encoder(_file(tmp_path, _state()), CPU, tokenizer_path=tmp_path / "tok")
    load_flux3_text_encoder(_file(tmp_path, _state()), CPU)

    assert encoder.tokenizer == "tok"
    assert seen == [str(tmp_path / "tok"), str(te.TOKENIZER_ASSETS)]


def test_a_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="Qwen3-VL-4B text encoder not found"):
        load_flux3_text_encoder(Path("absent.safetensors"), CPU)


@pytest.mark.parametrize("damage", ["missing", "unexpected"])
def test_a_file_that_is_not_the_expected_text_model_is_refused(tmp_path: Path, damage: str) -> None:
    state = _state()
    if damage == "missing":
        del state["layers.0.mlp.up_proj.weight"]
    else:
        state["layers.0.bogus.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match="does not look like a Qwen3-VL-4B text encoder"):
        load_flux3_text_encoder(_file(tmp_path, state), CPU)


def test_an_int8_file_runs_through_the_int8_linears_and_an_unrotating_embedding(tmp_path: Path) -> None:
    encoder = load_flux3_text_encoder(_file(tmp_path, _int8_state()), CPU)

    model = encoder.model
    out = encoder.encode("a cat", CPU)

    assert isinstance(model.embed_tokens, Int8Embedding)
    assert int(model.embed_tokens.int8_convrot_groupsize) == 16
    assert model.layers[0].self_attn.q_proj.weight.dtype == torch.int8
    assert out.shape == (80, 48) and torch.isfinite(out.float()).all()
