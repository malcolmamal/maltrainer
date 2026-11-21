from __future__ import annotations

from typing import cast

import pytest
import torch

from maltrainer.models.flux3.lora import LoRALayer
from maltrainer.models.ltx25.lora import LTX25LoRANetwork, param_group_of, selected

from .tiny_model import tiny_inputs, tiny_transformer

# per block: 6 attentions of 4 Linears (4 video, 4 audio, 8 cross-modal) and 2 + 2 feed-forward Linears
BLOCKS = 2


@pytest.mark.parametrize(
    ("targets", "count"),
    [("attn", BLOCKS * 24), ("video", BLOCKS * 8), ("attn_ff", BLOCKS * 28), ("video_ff", BLOCKS * 10), ("full", None)],
)
def test_each_preset_adapts_the_expected_number_of_linears(targets: str, count: int | None) -> None:
    model = tiny_transformer()
    network = LTX25LoRANetwork(model, rank=2, alpha=2.0, targets=targets)

    if count is None:
        count = sum(
            1
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.startswith("transformer_blocks.")
        )
    assert len(network.lora_layers) == len(network.name_map) == count


def test_the_video_preset_leaves_the_audio_and_cross_modal_layers_and_the_connectors_alone() -> None:
    names = set(LTX25LoRANetwork(tiny_transformer(), rank=2, targets="video").name_map.values())

    assert "transformer_blocks.0.attn1.to_q" in names and "transformer_blocks.1.attn2.to_out.0" in names
    assert not any("audio" in name or "connector" in name or ".ff." in name for name in names)


def test_the_attn_preset_reaches_every_attention_including_the_cross_modal_ones() -> None:
    names = set(LTX25LoRANetwork(tiny_transformer(), rank=2, targets="attn").name_map.values())

    assert {
        "transformer_blocks.0.audio_attn2.to_v",
        "transformer_blocks.1.audio_to_video_attn.to_k",
        "transformer_blocks.1.video_to_audio_attn.to_out.0",
    } <= names
    assert not any(".ff." in name or ".audio_ff." in name for name in names)


def test_the_feed_forward_presets_adapt_both_of_its_linears() -> None:
    names = set(LTX25LoRANetwork(tiny_transformer(), rank=2, targets="video_ff").name_map.values())

    assert {"transformer_blocks.0.ff.net.0.proj", "transformer_blocks.0.ff.net.2"} <= names
    assert not any("audio_ff" in name for name in names)


@pytest.mark.parametrize("targets", ["attn", "video", "attn_ff", "full"])
def test_a_fresh_network_leaves_the_model_output_unchanged(targets: str) -> None:
    model = tiny_transformer()
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    before = model(video, sigma, features, 24.0, audio, audio_sigma)

    network = LTX25LoRANetwork(model, rank=4, alpha=8.0, targets=targets)
    after = network(video, sigma, features, 24.0, audio, audio_sigma)

    assert before[1] is not None and after[1] is not None
    assert torch.equal(after[0], before[0]) and torch.equal(after[1], before[1])


def test_a_trained_adapter_shifts_the_output_and_receives_a_gradient() -> None:
    model = tiny_transformer().requires_grad_(False)
    video, sigma, features, audio, audio_sigma = tiny_inputs()
    before = model(video, sigma, features, 24.0, audio, audio_sigma)[0]
    network = LTX25LoRANetwork(model, rank=2, alpha=2.0)
    layer = cast(LoRALayer, network.lora_layers["transformer_blocks_0_attn1_to_q"])
    layer.lora_up.weight.data.normal_(std=0.5)

    after, audio_after = network(video, sigma, features, 24.0, audio, audio_sigma)
    assert audio_after is not None
    (after.square().sum() + audio_after.square().sum()).backward()

    assert not torch.allclose(after, before)
    assert layer.lora_down.weight.grad is not None and all(p.grad is None for p in model.parameters())


def test_the_state_dict_holds_comfyuis_keys_with_the_alpha() -> None:
    network = LTX25LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    state = network.get_state_dict(torch.float16)

    assert state["diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"].shape == (2, 96)
    assert state["diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight"].shape == (96, 2)
    assert state["diffusion_model.transformer_blocks.0.audio_attn1.to_q.lora_B.weight"].dtype == torch.float16
    assert state["diffusion_model.transformer_blocks.1.attn2.to_k.alpha"].item() == 4.0
    assert len(state) == 3 * len(network.lora_layers)


def test_the_prefix_and_the_unconverted_dtype_are_honoured() -> None:
    state = LTX25LoRANetwork(tiny_transformer(), rank=2, prefix="transformer.").get_state_dict(None)

    assert state["transformer.transformer_blocks.0.attn1.to_q.lora_A.weight"].dtype == torch.float32


def test_the_optimizer_groups_split_into_attention_mlp_and_other() -> None:
    network = LTX25LoRANetwork(tiny_transformer(), rank=2, targets="full")

    groups = {g["name"]: g["params"] for g in network.param_groups(split=True)}
    whole = network.param_groups(split=False)

    assert set(groups) == {"attn", "mlp", "other"}
    assert sum(len(p) for p in groups.values()) == len(whole[0]["params"]) == 2 * len(network.lora_layers)
    assert whole[0]["name"] == "lora"


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("transformer_blocks.0.attn1.to_q", "attn"),
        ("transformer_blocks.0.audio_to_video_attn.to_out.0", "attn"),
        ("transformer_blocks.0.ff.net.0.proj", "mlp"),
        ("transformer_blocks.0.audio_ff.net.2", "mlp"),
        ("patchify_proj", "other"),
    ],
)
def test_modules_group_by_family(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_the_lora_dtype_is_applied_to_the_adapters_only() -> None:
    model = tiny_transformer()
    network = LTX25LoRANetwork(model, rank=2)

    network.set_lora_dtype("cpu", torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_an_unknown_preset_is_refused() -> None:
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        LTX25LoRANetwork(tiny_transformer(), targets="everything")
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        selected("transformer_blocks.0.attn1.to_q", "everything")


def test_a_model_without_matching_linears_is_refused() -> None:
    with pytest.raises(ValueError, match=r"No LTX-2\.5 modules matched"):
        LTX25LoRANetwork(torch.nn.Sequential(torch.nn.Linear(2, 2)), rank=2)
