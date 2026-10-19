from __future__ import annotations

from typing import Callable, Dict, List

import pytest
import torch

from maltrainer.models.moss_image.lora import (
    DEFAULT_SKIP_BLOCKS,
    TARGET_PRESETS,
    MOSSImageLoRANetwork,
    param_group_of,
    selected,
)
from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]


@pytest.mark.parametrize(
    ("name", "blocks", "attn", "everything"),
    [
        ("layers.3.proj.q", True, True, True),
        ("layers.3.proj.o", True, True, True),
        ("layers.3.proj.ffn_up_gate", True, False, True),
        ("layers.3.proj.ffn_down", True, False, True),
        ("layers.3.proj.qkv", False, False, False),
        ("layers.3.q_norm", False, False, False),
        ("video_embedder", False, False, True),
        ("time_embedding.2", False, False, True),
        ("final_linear_video", False, False, True),
        ("text_embedder.proj", False, False, False),
        ("text_embedder.blocks.0.qkv", False, False, False),
        ("text_embedder.blocks.0.ffn_up", False, False, False),
    ],
)
def test_selection_by_preset_never_includes_the_text_side(
    name: str, blocks: bool, attn: bool, everything: bool
) -> None:
    assert [selected(name, preset) for preset in ("blocks", "attn", "all")] == [blocks, attn, everything]


def test_selection_rejects_an_unknown_preset() -> None:
    with pytest.raises(ValueError, match="lora.targets must be one of"):
        selected("layers.0.proj.q", "text")

    assert TARGET_PRESETS == ("blocks", "attn", "all")


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("layers.0.proj.q", "attn"),
        ("layers.41.proj.o", "attn"),
        ("layers.7.proj.ffn_up_gate", "mlp"),
        ("layers.7.proj.ffn_down", "mlp"),
        ("text_embedder.blocks.1.qkv", "attn"),
        ("text_embedder.blocks.1.proj", "attn"),
        ("text_embedder.blocks.0.ffn_gate", "mlp"),
        ("text_embedder.proj", "other"),
        ("video_embedder", "other"),
        ("shared_block_modulation", "other"),
    ],
)
def test_each_module_family_gets_its_own_automagic_group(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_no_block_is_left_without_an_adapter_by_default() -> None:
    assert DEFAULT_SKIP_BLOCKS == ()


def test_one_group_holds_every_adapter_when_the_groups_are_not_split(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(), rank=2)

    (group,) = network.param_groups(split=False)

    assert group["name"] == "lora"
    assert group["params"] == list(network.lora_layers.parameters())


def test_split_groups_partition_the_adapters_by_family(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(), rank=2, targets="all")

    groups = {g["name"]: g["params"] for g in network.param_groups(split=True)}

    assert sorted(groups) == ["attn", "mlp", "other"]
    assert [len(groups[n]) for n in ("attn", "mlp", "other")] == [16, 8, 12]
    flat = [p for params in groups.values() for p in params]
    assert len(flat) == len({id(p) for p in flat}) == len(list(network.trainable_parameters()))


def test_the_state_dict_uses_comfyuis_keys_with_the_prefix_and_a_float32_alpha(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(fused_qkv=True), rank=2, alpha=5.0, targets="attn", prefix="x.")

    state = network.get_state_dict()

    assert sorted(state) == sorted(
        f"x.layers.{i}.proj.{leaf}.{suffix}"
        for i in (0, 1)
        for leaf in "qkvo"
        for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
    )
    assert state["x.layers.0.proj.k.lora_A.weight"].shape == (2, 128)
    assert state["x.layers.0.proj.k.lora_B.weight"].shape == (128, 2)
    assert state["x.layers.0.proj.k.alpha"].dtype == torch.float32
    assert state["x.layers.0.proj.k.alpha"].item() == 5.0
    assert state["x.layers.0.proj.k.alpha"].dim() == 0


def test_the_default_prefix_is_the_diffusion_model_container(make_model: Maker) -> None:
    state = MOSSImageLoRANetwork(make_model(), rank=2).get_state_dict()

    assert "diffusion_model.layers.1.proj.ffn_down.lora_A.weight" in state


@pytest.mark.parametrize(
    ("save_dtype", "expected"),
    [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16), (None, torch.float32)],
)
def test_weights_are_saved_in_the_requested_dtype_and_alpha_always_in_float32(
    make_model: Maker, save_dtype: torch.dtype | None, expected: torch.dtype
) -> None:
    network = MOSSImageLoRANetwork(make_model(), rank=2)

    state = network.get_state_dict(save_dtype)

    dtypes: Dict[str, List[torch.dtype]] = {}
    for key, value in state.items():
        dtypes.setdefault(key.rsplit(".", 1)[-1], []).append(value.dtype)
    assert set(dtypes["weight"]) == {expected}
    assert set(dtypes["alpha"]) == {torch.float32}


def test_saved_weights_hold_the_trained_values_without_a_graph(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(), rank=2)
    network.lora_layers["layers_0_proj_q"].lora_up.weight.data.fill_(0.25)

    saved = network.get_state_dict(None)["diffusion_model.layers.0.proj.q.lora_B.weight"]

    assert not saved.requires_grad
    assert saved.device.type == "cpu"
    assert saved.eq(0.25).all()


def test_set_lora_dtype_converts_every_adapter(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(), rank=2)

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
