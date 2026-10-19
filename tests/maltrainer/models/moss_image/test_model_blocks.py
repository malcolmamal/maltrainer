from __future__ import annotations

import pytest
import torch

from maltrainer.models.moss_image.model import (
    Block,
    MOSSImageConfig,
    MOSSImageTransformer,
    TextBlock,
    position_ids,
    rope_from_ids,
)


def _meta_state(config: MOSSImageConfig) -> dict:
    with torch.device("meta"):
        return MOSSImageTransformer(config).state_dict()


@pytest.mark.parametrize("fused_qkv", [False, True])
def test_the_default_config_is_what_the_shipped_checkpoints_read_back_as(fused_qkv: bool) -> None:
    config = MOSSImageConfig(fused_qkv=fused_qkv)

    assert MOSSImageConfig.from_state_dict(_meta_state(config)) == config


def test_from_state_dict_reads_heads_as_dim_over_128_and_counts_the_distinct_layer_indices() -> None:
    config = MOSSImageConfig(dim=256, ffn_dim=24, num_heads=2, num_layers=3, text_dim=5)

    read = MOSSImageConfig.from_state_dict(_meta_state(config))

    assert (read.dim, read.ffn_dim, read.num_heads, read.num_layers, read.text_dim) == (256, 24, 2, 3, 5)


def test_from_state_dict_rejects_a_checkpoint_that_is_not_a_moss_image_dit() -> None:
    with pytest.raises(ValueError, match="Not a MOSS-Image DiT checkpoint"):
        MOSSImageConfig.from_state_dict({"layers.0.proj.q.weight": torch.empty(1)})


def test_block_parameters_are_named_as_comfyuis_unfused_and_fused_files_name_them() -> None:
    unfused = {k for k in Block(8, 4, 1, fused_qkv=False).state_dict()}
    fused = {k for k in Block(8, 4, 1, fused_qkv=True).state_dict()}

    common = {"norm1.weight", "norm2.weight", "q_norm.weight", "k_norm.weight"}
    ffn = {"proj.ffn_up_gate.weight", "proj.ffn_down.weight", "proj.o.weight"}
    assert unfused == common | ffn | {f"proj.{n}.weight" for n in "qkv"}
    assert fused == common | ffn | {"proj.qkv.weight"}


def test_a_fused_block_computes_what_an_unfused_one_with_the_same_weights_computes() -> None:
    torch.manual_seed(1)
    unfused = Block(128, 16, 1, fused_qkv=False)
    fused = Block(128, 16, 1, fused_qkv=True)
    state = unfused.state_dict()
    qkv = torch.cat([state[f"proj.{n}.weight"] for n in "qkv"])
    fused.load_state_dict({**{k: v for k, v in state.items() if ".q." not in k}, "proj.qkv.weight": qkv}, strict=False)
    x = torch.randn(1, 6, 128)
    mods = [torch.randn(1, 1, 128) for _ in range(6)]
    pe = rope_from_ids(position_ids(2, 2, 2, torch.device("cpu")), head_dim=128)

    out_fused, out_unfused = fused(x, *mods, pe), unfused(x, *mods, pe)

    torch.testing.assert_close(out_fused, out_unfused, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_fused, x)


def test_a_block_whose_gates_are_zero_returns_its_input() -> None:
    block = Block(128, 16, 1, fused_qkv=False)
    x = torch.randn(2, 6, 128)
    shift_scale = torch.randn(2, 1, 128)
    zero = torch.zeros(2, 1, 128)
    pe = rope_from_ids(position_ids(2, 2, 2, torch.device("cpu")), head_dim=128)

    out = block(x, shift_scale, shift_scale, zero, shift_scale, shift_scale, zero, pe)

    torch.testing.assert_close(out, x)


def test_a_block_adds_the_gated_attention_branch_to_the_residual() -> None:
    torch.manual_seed(2)
    block = Block(128, 16, 1, fused_qkv=False)
    x = torch.randn(1, 6, 128)
    zero, one = torch.zeros(1, 1, 128), torch.ones(1, 1, 128)
    pe = rope_from_ids(position_ids(2, 2, 2, torch.device("cpu")), head_dim=128)

    full = block(x, zero, zero, one, zero, zero, zero, pe)
    half = block(x, zero, zero, one * 0.5, zero, zero, zero, pe)

    torch.testing.assert_close(half - x, (full - x) * 0.5, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(full, x)


def test_a_text_block_is_a_plain_residual_with_biased_linears() -> None:
    torch.manual_seed(3)
    block = TextBlock(128, 16, 1)
    block.proj.weight.data.zero_()
    block.proj.bias.data.zero_()
    block.ffn_down.weight.data.zero_()
    block.ffn_down.bias.data.zero_()
    x = torch.randn(1, 4, 128)
    pe = rope_from_ids(position_ids(1, 1, 4, torch.device("cpu")), head_dim=128)[:, 1:]

    out = block(x, pe)

    torch.testing.assert_close(out, x)
    assert {"qkv.bias", "proj.bias", "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"} <= set(block.state_dict())


def test_the_transformer_state_dict_uses_comfyuis_module_names_and_shapes() -> None:
    state = _meta_state(MOSSImageConfig(dim=128, ffn_dim=16, num_heads=1, num_layers=2, text_dim=8))

    assert state["video_embedder.weight"].shape == (128, 128)
    assert state["text_embedder.proj.weight"].shape == (128, 8)
    assert state["text_embedder.blocks.1.qkv.bias"].shape == (384,)
    assert state["time_embedding.0.weight"].shape == (128, 256)
    assert state["time_embedding.2.bias"].shape == (128,)
    assert state["shared_block_modulation.weight"].shape == (6 * 128, 128)
    assert state["video_head_modulation.weight"].shape == (2 * 128, 128)
    assert state["layers.1.proj.ffn_up_gate.weight"].shape == (32, 128)
    assert state["layers.1.proj.ffn_down.weight"].shape == (128, 16)
    assert state["layers.0.q_norm.weight"].shape == (128,)
    assert state["final_linear_video.weight"].shape == (128, 128)
    assert "final_linear_video.bias" not in state
    assert not any(key.startswith("layers.") and key.endswith(".bias") for key in state)
