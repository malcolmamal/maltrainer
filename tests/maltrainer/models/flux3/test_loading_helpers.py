from __future__ import annotations

from pathlib import Path

import pytest
import torch

from maltrainer.models.flux3.loading import _strip_container_prefix, _to_module_names, is_int8_convrot
from tests.maltrainer.models.flux3.checkpoints import marker, plain_state, write


def _hint(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def test_a_file_with_comfy_quant_markers_is_an_int8_checkpoint(tmp_path: Path) -> None:
    int8 = write(tmp_path / "int8.safetensors", {"a.weight": torch.zeros(1), "a.comfy_quant": marker(format="x")})
    plain = write(tmp_path / "plain.safetensors", {"a.weight": torch.zeros(1)})

    assert is_int8_convrot(int8) is True
    assert is_int8_convrot(str(plain)) is False


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
def test_a_container_prefix_shared_by_every_key_is_stripped(prefix: str) -> None:
    state = {f"{prefix}single_blocks.0.q_proj.weight": torch.zeros(1), f"{prefix}txt_in.weight": torch.ones(1)}

    assert set(_strip_container_prefix(state)) == {"single_blocks.0.q_proj.weight", "txt_in.weight"}


@pytest.mark.parametrize(
    "keys",
    [[], ["diffusion_model.a", "b"], ["model.a", "model.b"]],
)
def test_keys_without_a_shared_container_prefix_are_left_alone(keys: list) -> None:
    state = {key: torch.zeros(1) for key in keys}

    assert _strip_container_prefix(state) is state


def test_qk_norm_scales_are_renamed_to_module_weights_and_everything_else_is_kept() -> None:
    state = {
        "single_blocks.0.norm.query_norm.scale": torch.ones(2),
        "single_blocks.0.norm.key_norm.scale": torch.zeros(2),
        "single_blocks.0.q_proj.weight": torch.zeros(2, 2),
    }

    renamed, hints = _to_module_names(state)

    assert set(renamed) == {
        "single_blocks.0.norm.query_norm.weight",
        "single_blocks.0.norm.key_norm.weight",
        "single_blocks.0.q_proj.weight",
    }
    assert torch.equal(renamed["single_blocks.0.norm.query_norm.weight"], torch.ones(2))
    assert hints == {}


def test_the_plain_test_state_round_trips_through_the_renaming() -> None:
    state = plain_state()

    renamed, _ = _to_module_names(state)

    assert not any(key.endswith("_norm.scale") for key in renamed)
    assert len(renamed) == len(state)


def test_attention_hints_are_counted_by_method_and_dropped_from_the_weights() -> None:
    state = {
        "single_blocks.0.comfy_attention.config": _hint('{"attention": "int8"}'),
        "single_blocks.1.comfy_attention.config": _hint('{"attention": "int8"}'),
        "single_blocks.2.comfy_attention.config": _hint('[{"attention": "sol"}, {"attention": "int8"}]'),
        "txt_in.weight": torch.zeros(1),
    }

    renamed, hints = _to_module_names(state)

    assert list(renamed) == ["txt_in.weight"]
    assert hints == {"int8": "2 blocks", "sol,int8": "1 blocks"}


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"just a string"'])
def test_a_hint_that_cannot_be_read_is_reported_as_unreadable(text: str) -> None:
    _, hints = _to_module_names({"a.comfy_attention.config": _hint(text)})

    assert hints == {"unreadable": "1 blocks"}
