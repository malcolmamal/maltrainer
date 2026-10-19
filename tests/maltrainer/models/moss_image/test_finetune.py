from __future__ import annotations

from typing import Callable

import pytest
import torch
from torch import nn

from maltrainer.models.moss_image.finetune import (
    DEFAULT_TRAINABLE_PREFIXES,
    DRIFT_BLOCKS,
    EXTRACT_PROFILE,
    drift_targets,
    param_group_of,
    param_groups,
)
from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("layers.0.proj.q.weight", "attn"),
        ("layers.5.proj.qkv.weight", "attn"),
        ("layers.5.proj.o.weight", "attn"),
        ("layers.5.proj.ffn_up_gate.weight", "mlp"),
        ("layers.5.proj.ffn_down.weight", "mlp"),
        ("layers.5.norm1.weight", "norm"),
        ("layers.5.q_norm.weight", "norm"),
        ("text_embedder.blocks.0.qkv.weight", "attn"),
        ("text_embedder.blocks.0.proj.bias", "attn"),
        ("text_embedder.blocks.1.ffn_gate.weight", "mlp"),
        ("text_embedder.blocks.1.norm2.weight", "norm"),
        ("text_embedder.proj.weight", "other"),
        ("video_embedder.weight", "other"),
        ("final_linear_video.weight", "other"),
    ],
)
def test_each_parameter_family_gets_its_own_automagic_group(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_the_default_trainable_set_is_the_joint_blocks_only() -> None:
    assert DEFAULT_TRAINABLE_PREFIXES == ("layers.",)


def test_unsplit_parameters_form_one_group_in_order() -> None:
    named = [
        ("layers.0.proj.q.weight", nn.Parameter(torch.zeros(1))),
        ("video_embedder.weight", nn.Parameter(torch.zeros(1))),
    ]

    (group,) = param_groups(named, split=False)

    assert group["name"] == "all"
    assert group["params"] == [p for _, p in named]


def test_split_parameters_are_grouped_by_family_keeping_every_parameter_once() -> None:
    named = [
        (n, nn.Parameter(torch.zeros(1)))
        for n in (
            "layers.0.proj.q.weight",
            "layers.0.proj.ffn_down.weight",
            "layers.0.proj.o.weight",
            "layers.0.norm1.weight",
        )
    ]

    groups = {g["name"]: g["params"] for g in param_groups(named, split=True)}

    assert list(groups) == ["attn", "mlp", "norm"]
    assert groups["attn"] == [named[0][1], named[2][1]]
    assert groups["mlp"] == [named[1][1]]
    assert groups["norm"] == [named[3][1]]


def test_drift_targets_pick_the_attention_output_and_ffn_down_of_the_five_tracked_blocks() -> None:
    leaves = ("proj.q.weight", "proj.o.weight", "proj.ffn_down.weight", "proj.ffn_up_gate.weight", "norm1.weight")
    named = [(f"layers.{i}.{leaf}", nn.Parameter(torch.zeros(1))) for i in range(42) for leaf in leaves]
    named.append(("text_embedder.blocks.0.proj.weight", nn.Parameter(torch.zeros(1))))

    targets = drift_targets(named)

    assert DRIFT_BLOCKS == (0, 10, 21, 31, 41)
    assert [label for label, _ in targets] == ["attn", "mlp"] * 5
    wanted = {f"layers.{i}.{leaf}" for i in DRIFT_BLOCKS for leaf in ("proj.o.weight", "proj.ffn_down.weight")}
    assert {id(p) for _, p in targets} == {id(p) for n, p in named if n in wanted}


def test_a_model_too_small_to_have_tracked_blocks_has_nothing_to_track(make_model: Maker) -> None:
    named = [(n, p) for n, p in make_model(num_layers=2).named_parameters() if n.startswith("layers.1.")]

    assert [label for label, _ in drift_targets(named)] == []


def test_the_extract_profile_splits_a_fused_qkv_into_q_k_v_thirds() -> None:
    diff = torch.arange(12.0).reshape(6, 2)

    parts = EXTRACT_PROFILE.split("layers.3.proj.qkv", diff)

    assert [name for name, _ in parts] == ["layers.3.proj.q", "layers.3.proj.k", "layers.3.proj.v"]
    assert [chunk.tolist() for _, chunk in parts] == [diff[:2].tolist(), diff[2:4].tolist(), diff[4:].tolist()]


@pytest.mark.parametrize("name", ["layers.3.proj.o", "text_embedder.blocks.0.qkv", "video_embedder"])
def test_the_extract_profile_leaves_every_other_module_whole(name: str) -> None:
    diff = torch.ones(6, 2)

    ((out_name, out_diff),) = EXTRACT_PROFILE.split(name, diff)

    assert out_name == name
    assert out_diff is diff


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"], True),
        (["shared_block_modulation.weight", "layers.0.proj.q.weight"], False),
        (["layers.0.proj.ffn_up_gate.weight"], False),
        (["transformer_blocks.0.attn.to_q.weight"], False),
    ],
)
def test_the_extract_profile_recognises_a_moss_image_checkpoint(names: list, expected: bool) -> None:
    assert EXTRACT_PROFILE.detect(names) is expected
    assert EXTRACT_PROFILE.name == "moss_image"
