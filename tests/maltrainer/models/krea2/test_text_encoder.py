"""Krea 2 text conditioning: the 12-layer hidden-state tap and the checkpoint loader, on a tiny Qwen3-VL."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from maltrainer.models.krea2 import text_encoder as te
from maltrainer.models.krea2.text_encoder import Krea2TextEncoder, _prefix_end, load_krea2_text_encoder

from .tiny import CPU
from .tiny_text import TAPS, TINY_TEXT, tiny_text_model


@pytest.fixture(autouse=True)
def tiny_encoder_settings(monkeypatch):
    monkeypatch.setattr(te, "TAP_LAYERS", TAPS)
    monkeypatch.setattr(te, "QWEN3VL_4B_TEXT", TINY_TEXT)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS), trust_remote_code=True)


def test_encode_stacks_the_tapped_hidden_states_and_drops_only_the_template_prefix(tokenizer):
    model = tiny_text_model()
    encoder = Krea2TextEncoder(model, tokenizer)
    out = encoder.encode("a cat", CPU, dtype=torch.float32)
    ids = tokenizer(te.TEMPLATE.format("a cat"), add_special_tokens=False).input_ids
    start = _prefix_end(ids)
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    assert out.shape == (len(ids) - start, 3, 16)
    for position, layer in enumerate(TAPS):
        assert torch.allclose(out[:, position], hidden[layer][0, start:], atol=1e-5), layer


def test_encode_returns_the_requested_dtype(tokenizer):
    encoder = Krea2TextEncoder(tiny_text_model(), tokenizer)
    assert encoder.encode("a cat", CPU, dtype=torch.bfloat16).dtype == torch.bfloat16


def test_prompt_tokens_still_attend_to_the_dropped_prefix(tokenizer):
    model = tiny_text_model()
    encoder = Krea2TextEncoder(model, tokenizer)
    full = encoder.encode("a cat", CPU, dtype=torch.float32)
    ids = tokenizer("a cat", add_special_tokens=False).input_ids
    bare = model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]
    assert not torch.allclose(full[: len(ids), -1], bare, atol=1e-4)


def test_encode_is_deterministic_and_prompt_dependent(tokenizer):
    encoder = Krea2TextEncoder(tiny_text_model(), tokenizer)
    cat = encoder.encode("a cat", CPU, dtype=torch.float32)
    assert torch.equal(cat, encoder.encode("a cat", CPU, dtype=torch.float32))
    dog = encoder.encode("a dog", CPU, dtype=torch.float32)
    assert dog.shape == cat.shape and not torch.allclose(dog, cat)


def test_prompts_longer_than_the_cap_are_cut_to_that_many_tokens(tokenizer):
    encoder = Krea2TextEncoder(tiny_text_model(), tokenizer)
    prompt = "one two three four five six"
    cut = encoder.encode(prompt, CPU, dtype=torch.float32, max_prompt_tokens=3)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    expected = encoder.encode(tokenizer.decode(ids[:3]), CPU, dtype=torch.float32)
    assert torch.equal(cut, expected)
    assert cut.shape[0] == encoder.encode(prompt, CPU, dtype=torch.float32).shape[0] - (len(ids) - 3)


@pytest.fixture
def encoder_file(tmp_path):
    model = tiny_text_model()
    state = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    state["model.norm.weight"] = torch.ones(16)  # the final norm is replaced by an Identity
    state["model.layers.3.self_attn.q_proj.weight"] = torch.zeros(2)  # beyond the truncated depth
    state["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(2)
    state["lm_head.weight"] = torch.zeros(2)
    path = tmp_path / "qwen3vl_4b.safetensors"
    save_file(state, str(path))
    return model, path


def test_loader_builds_a_truncated_encoder_that_matches_the_checkpoint_weights(encoder_file):
    model, path = encoder_file
    encoder = load_krea2_text_encoder(path, CPU, dtype=torch.float32)
    assert len(encoder.model.layers) == 3 and isinstance(encoder.model.norm, torch.nn.Identity)
    assert not any(p.requires_grad for p in encoder.model.parameters()) and not encoder.model.training
    reference = Krea2TextEncoder(model, encoder.tokenizer).encode("a cat", CPU, dtype=torch.float32)
    assert torch.allclose(encoder.encode("a cat", CPU, dtype=torch.float32), reference, atol=1e-5)


def test_loader_casts_to_the_requested_dtype(encoder_file):
    encoder = load_krea2_text_encoder(encoder_file[1], CPU, dtype=torch.bfloat16)
    assert encoder.model.embed_tokens.weight.dtype == torch.bfloat16


def test_loader_refuses_a_file_that_is_not_a_qwen3_vl_text_encoder(tmp_path, encoder_file):
    model, _ = encoder_file
    state = {f"model.{k}": v for k, v in model.state_dict().items() if "q_proj" not in k}
    path = tmp_path / "bad.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    with pytest.raises(ValueError, match="does not look like a Qwen3-VL-4B text encoder"):
        load_krea2_text_encoder(path, CPU)
