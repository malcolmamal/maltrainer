from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.flux3.model import Block, LastLayer, MLPEmbedder, Modulation, QKNorm, RMSNorm, rope_from_ids

D = 128


def _block(fused: bool = False, seed: int = 0) -> Block:
    torch.manual_seed(seed)
    return Block(D, num_heads=1, mlp_ratio=1.0, fused_qkv=fused).eval()


def _pe(tokens: int) -> torch.Tensor:
    ids = torch.zeros(tokens, 4)
    ids[:, 3] = torch.arange(tokens)
    return rope_from_ids(ids)


def _mod(batch: int, value: float, rows: int = 1) -> torch.Tensor:
    return torch.full((batch, rows, D), value)


def test_the_rms_norm_scales_by_the_root_mean_square_and_its_weight() -> None:
    norm = RMSNorm(2)
    norm.weight.data = torch.tensor([2.0, 1.0])

    out = norm(torch.tensor([[3.0, 4.0]]))

    rms = math.sqrt(12.5)
    torch.testing.assert_close(out, torch.tensor([[6.0 / rms, 4.0 / rms]]), atol=1e-5, rtol=0)


def test_the_rms_norm_returns_the_dtype_it_was_given() -> None:
    out = RMSNorm(4)(torch.ones(2, 4, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16


def test_qk_norm_holds_separate_query_and_key_scales() -> None:
    names = {name for name, _ in QKNorm(8).named_parameters()}

    assert names == {"query_norm.weight", "key_norm.weight"}


def test_the_mlp_embedder_is_a_bias_free_two_layer_silu_mlp() -> None:
    embedder = MLPEmbedder(1, 1)
    embedder.in_layer.weight.data.fill_(2.0)
    embedder.out_layer.weight.data.fill_(3.0)

    out = embedder(torch.tensor([[1.0]]))

    torch.testing.assert_close(out, torch.tensor([[3.0 * F.silu(torch.tensor(2.0))]]))
    assert [n for n, _ in embedder.named_parameters()] == ["in_layer.weight", "out_layer.weight"]


def test_a_modulation_splits_silu_of_the_vector_into_shift_scale_and_gate() -> None:
    torch.manual_seed(0)
    modulation = Modulation(4)
    vec = torch.randn(2, 4)

    shift, scale, gate = modulation(vec)

    full = modulation.lin(F.silu(vec[:, None, :]))
    assert shift.shape == scale.shape == gate.shape == (2, 1, 4)
    torch.testing.assert_close(torch.cat([shift, scale, gate], dim=-1), full)


def test_a_block_with_a_closed_gate_passes_its_input_through() -> None:
    block = _block()
    x = torch.randn(2, 3, D)

    out = block(x, _mod(2, 0.3), _mod(2, 0.1), _mod(2, 0.0), _pe(3))

    torch.testing.assert_close(out, x)


def test_a_block_without_attention_adds_the_gated_swiglu_of_the_modulated_input() -> None:
    block = _block()
    block.attn_out.weight.data.zero_()
    x = torch.randn(1, 2, D)
    shift, scale, gate = _mod(1, 0.5), _mod(1, 0.25), _mod(1, 2.0)

    out = block(x, shift, scale, gate, _pe(2))

    x_mod = F.layer_norm(x, (D,), eps=1e-6) * 1.25 + 0.5
    gate_part, up = block.mlp_in(x_mod).chunk(2, dim=-1)
    torch.testing.assert_close(out, x + 2.0 * block.mlp_out(F.silu(gate_part) * up))


def test_a_one_token_block_without_an_mlp_adds_the_output_projection_of_the_value() -> None:
    block = _block()
    block.mlp_out.weight.data.zero_()
    x = torch.randn(2, 1, D)

    out = block(x, _mod(2, 0.0), _mod(2, 0.0), _mod(2, 1.0), _pe(1))

    x_mod = F.layer_norm(x, (D,), eps=1e-6)
    torch.testing.assert_close(out, x + block.attn_out(block.v_proj(x_mod)), atol=1e-5, rtol=1e-5)


def test_a_fused_qkv_block_matches_the_split_block_holding_the_same_weights() -> None:
    split, fused = _block(fused=False), _block(fused=True, seed=1)
    state = {k: v for k, v in split.state_dict().items() if not k.startswith(("q_proj", "k_proj", "v_proj"))}
    state["qkv.weight"] = torch.cat([split.q_proj.weight, split.k_proj.weight, split.v_proj.weight])
    fused.load_state_dict(state)
    x = torch.randn(2, 5, D)
    args = (_mod(2, 0.2), _mod(2, -0.3), _mod(2, 0.7), _pe(5))

    torch.testing.assert_close(fused(x, *args), split(x, *args))


def test_a_per_row_modulation_equal_in_every_row_matches_a_per_mode_one() -> None:
    block = _block()
    x = torch.randn(2, 4, D)
    per_mode = (_mod(2, 0.2), _mod(2, -0.3), _mod(2, 0.7))
    per_row = tuple(_mod(2, m[0, 0, 0].item(), rows=4) for m in per_mode)

    torch.testing.assert_close(block(x, *per_row, _pe(4)), block(x, *per_mode, _pe(4)))


def test_rope_changes_which_tokens_attend_to_each_other() -> None:
    block = _block()
    x = torch.randn(1, 3, D)
    args = (_mod(1, 0.0), _mod(1, 0.0), _mod(1, 1.0))
    still = rope_from_ids(torch.zeros(3, 4))

    assert not torch.allclose(block(x, *args, _pe(3)), block(x, *args, still))


@pytest.mark.parametrize("fused", [False, True])
def test_a_block_names_its_projections_the_way_comfyui_does(fused: bool) -> None:
    names = {n.rsplit(".", 1)[0] for n, _ in _block(fused=fused).named_parameters()}

    projections = {"qkv"} if fused else {"q_proj", "k_proj", "v_proj"}
    assert names == projections | {"mlp_in", "attn_out", "mlp_out", "norm.query_norm", "norm.key_norm"}


def test_the_last_layer_projects_the_normalised_tokens_to_the_latent_channels() -> None:
    torch.manual_seed(0)
    layer = LastLayer(8, 6)
    layer.adaLN_modulation[1].weight.data.zero_()
    x = torch.randn(2, 3, 8)

    out = layer(x, torch.randn(2, 8))

    assert out.shape == (2, 3, 6)
    torch.testing.assert_close(out, layer.linear(F.layer_norm(x, (8,), eps=1e-6)))
