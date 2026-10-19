from __future__ import annotations

import pytest
import torch

from maltrainer.models.flux3 import model as flux3_model
from maltrainer.models.flux3.model import Flux3Config, Flux3Transformer, timestep_embedding
from tests.maltrainer.models.flux3.tiny_model import TINY, tiny_inputs, tiny_transformer


def _backward(model: Flux3Transformer) -> dict:
    model.zero_grad()
    model(*tiny_inputs()).square().sum().backward()
    return {name: p.grad for name, p in model.named_parameters()}


def test_the_velocity_has_the_shape_of_the_latent_and_the_dtype_of_the_compute_dtype() -> None:
    x, sigma, context = tiny_inputs(batch=2, h=2, w=3)

    out = tiny_transformer()(x, sigma, context)

    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_a_bf16_model_runs_in_bf16_whatever_the_input_dtype() -> None:
    model = tiny_transformer().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16

    out = model(*tiny_inputs())

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_samples_in_a_batch_do_not_influence_each_other() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs(batch=2)

    both = model(x, sigma, context)

    for i in range(2):
        torch.testing.assert_close(
            model(x[i : i + 1], sigma[i : i + 1], context[i : i + 1]), both[i : i + 1], atol=1e-5, rtol=1e-4
        )


def test_one_prompt_is_shared_by_every_sample_of_the_batch() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs(batch=2)

    shared = model(x, sigma, context[:1])

    torch.testing.assert_close(shared, model(x, sigma, context[:1].expand(2, -1, -1).contiguous()))


def test_the_output_depends_on_the_noise_level_and_on_the_prompt() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()

    base = model(x, sigma, context)

    assert not torch.allclose(base, model(x, sigma.flip(0), context))
    assert not torch.allclose(base, model(x, sigma, context.flip(0)))


def test_the_text_stream_is_embedded_at_t_zero_and_the_image_at_the_real_t() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs(batch=2)
    seen = {}
    model.time_in.register_forward_hook(lambda _m, args, out: seen.update(steps=args[0], vec=out))
    model.single_stream_modulations["txt"].register_forward_hook(lambda _m, args, _out: seen.update(txt_vec=args[0]))

    model(x, sigma, context)

    torch.testing.assert_close(seen["steps"][:-1], timestep_embedding(sigma, 256))
    torch.testing.assert_close(seen["steps"][-1:], timestep_embedding(torch.zeros(1), 256))
    torch.testing.assert_close(seen["txt_vec"], seen["vec"][-1:])


def test_only_the_text_to_image_modules_receive_gradients() -> None:
    grads = _backward(tiny_transformer())

    unused = {name for name, grad in grads.items() if grad is None}
    assert unused == {name for name in grads if "image_cond" in name}
    assert grads["emb_in.image.weight"] is not None
    assert grads["txt_mode_blocks.0.q_proj.weight"] is not None
    assert grads["single_blocks.1.mlp_out.weight"] is not None


def test_gradient_checkpointing_changes_the_schedule_but_not_the_gradients() -> None:
    plain = _backward(tiny_transformer())
    model = tiny_transformer()
    model.enable_gradient_checkpointing()

    checkpointed = _backward(model)

    assert model.gradient_checkpointing
    for name, grad in plain.items():
        if grad is not None:
            torch.testing.assert_close(checkpointed[name], grad, atol=1e-5, rtol=1e-4)


def test_every_block_is_checkpointed_only_while_gradients_are_on(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def counting(block, *args, use_reentrant):
        assert use_reentrant is False
        calls.append(block)
        return block(*args)

    monkeypatch.setattr(flux3_model, "checkpoint", counting)
    model = tiny_transformer()
    model.enable_gradient_checkpointing()
    inputs = tiny_inputs()

    with torch.no_grad():
        model(*inputs)
    assert calls == []
    model(*inputs)

    assert len(calls) == 1 + 1 + 2


@pytest.mark.parametrize(
    ("config", "present", "absent"),
    [
        (TINY, "single_blocks.1.q_proj.weight", {"ref_time_in.in_layer.weight", "single_blocks.0.qkv.weight"}),
        (Flux3Config(128, 1, 1, 1, fused_qkv=True), "single_blocks.0.qkv.weight", {"single_blocks.0.q_proj.weight"}),
        (Flux3Config(128, 1, 1, 1, flow_map=True), "ref_time_in.out_layer.weight", set()),
    ],
)
def test_the_parameters_carry_comfyuis_module_names(config: Flux3Config, present: str, absent: set) -> None:
    keys = set(Flux3Transformer(config).state_dict())

    assert present in keys
    assert not absent & keys
    assert {f"{m}.lin.weight" for m in ("early_stream_modulations.txt", "single_stream_modulations.image")} <= keys
    assert {"txt_in.weight", "time_in.out_layer.weight", "final_layer.image.adaLN_modulation.1.weight"} <= keys
    assert {"emb_in.image.weight", "emb_in.image_cond.weight"} <= keys
    assert {"content_mode_blocks.image_cond.0.norm.key_norm.weight", "txt_mode_blocks.0.attn_out.weight"} <= keys
