from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.krea2.finetune import DEFAULT_ALWAYS_ON, DEFAULT_WINDOWS, always_on_linears, parse_windows
from tests.maltrainer.models.krea2.tiny_model import tiny_transformer


@pytest.mark.parametrize("config", [None, []])
def test_no_configured_windows_means_attention_then_the_three_mlp_matrices(config: object) -> None:
    windows = parse_windows(config)

    assert windows is DEFAULT_WINDOWS
    assert [name for name, _ in windows] == ["attention", "mlp.gate", "mlp.up", "mlp.down"]


def test_the_default_windows_partition_every_block_linear() -> None:
    names = [f"attn.{n}" for n in ("wq", "wk", "wv", "gate", "wo")] + [f"mlp.{n}" for n in ("gate", "up", "down")]

    owners = [[w for w, prefixes in DEFAULT_WINDOWS if leaf.startswith(prefixes)] for leaf in names]

    assert all(len(o) == 1 for o in owners)


def test_windows_can_be_written_as_mappings_or_pairs_with_one_prefix_or_a_list() -> None:
    windows = parse_windows(
        [
            {"name": "attention", "prefixes": ["attn."]},
            ["mlp", "mlp."],
            ("tail", ["mlp.down", "attn.wo"]),
            {"name": 7, "prefixes": [8]},
        ]
    )

    assert windows == (
        ("attention", ("attn.",)),
        ("mlp", ("mlp.",)),
        ("tail", ("mlp.down", "attn.wo")),
        ("7", ("8",)),
    )


def test_the_text_fusion_stack_is_always_trained() -> None:
    assert DEFAULT_ALWAYS_ON == ("txtfusion",)


def test_every_linear_under_the_always_on_module_becomes_trainable_weight_only() -> None:
    model = tiny_transformer().requires_grad_(False)

    found = always_on_linears(model, DEFAULT_ALWAYS_ON)

    names = [name for name, _ in found]
    assert len(found) == 4 * 8 + 1
    assert "txtfusion.projector.weight" in names
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight" in names
    assert all(param.requires_grad for _, param in found)
    assert found[0][1] is model.get_submodule(names[0][: -len(".weight")]).weight
    assert not model.txtfusion.layerwise_blocks[0].prenorm.scale.requires_grad
    assert not model.first.weight.requires_grad


def test_always_on_modules_outside_the_stack_are_found_by_their_own_names() -> None:
    model = tiny_transformer().requires_grad_(False)

    found = always_on_linears(model, ["last", "tproj"])

    assert [name for name, _ in found] == ["last.linear.weight", "tproj.1.weight"]
    assert model.last.linear.bias.requires_grad is False


def test_a_linear_holding_fp8_weights_cannot_be_always_on() -> None:
    model = nn.Sequential()
    model.add_module("txt", nn.Sequential(nn.Linear(2, 2)))
    model.txt[0].weight = nn.Parameter(torch.zeros(2, 2).to(torch.float8_e4m3fn), requires_grad=False)

    with pytest.raises(ValueError, match=r"txt\.0 is fp8; always-on modules must be bf16"):
        always_on_linears(model, ["txt"])
