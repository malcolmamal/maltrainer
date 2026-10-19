from __future__ import annotations

import torch

from maltrainer.models.qwen_image_2_1.model import QwenImage21Transformer, build_rope_frequencies

from ._tiny import TINY_CONFIG, tiny_inputs, tiny_transformer


def _forward(model: QwenImage21Transformer, batch: int = 2, h: int = 2, w: int = 3, text_len: int = 3) -> torch.Tensor:
    latents, text, timestep = tiny_inputs(batch, h, w, text_len)
    return model(latents, text, timestep, latent_h=h, latent_w=w)


def _capture_block_calls(model: QwenImage21Transformer) -> list:
    calls: list = []

    def record(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        calls.append((args, kwargs))

    model.transformer_blocks[0].register_forward_pre_hook(record, with_kwargs=True)
    return calls


def test_the_forward_returns_one_velocity_vector_per_image_token() -> None:
    out = _forward(tiny_transformer(), batch=2, h=2, w=3, text_len=3)

    assert out.shape == (2, 6, TINY_CONFIG["out_channels"])
    assert out.dtype == torch.float32


def test_the_forward_is_deterministic_for_a_seed_and_differs_between_seeds() -> None:
    first = _forward(tiny_transformer(seed=3))
    second = _forward(tiny_transformer(seed=3))
    other = _forward(tiny_transformer(seed=4))

    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.allclose(first, other)


def test_a_channels_first_latent_grid_is_flattened_to_the_same_tokens() -> None:
    model = tiny_transformer()
    latents, text, timestep = tiny_inputs(batch=2, h=2, w=3)
    grid = latents.reshape(2, 2, 3, 4).permute(0, 3, 1, 2)

    flat = model(latents, text, timestep, latent_h=2, latent_w=3)
    gridded = model(grid, text, timestep, latent_h=2, latent_w=3)

    torch.testing.assert_close(gridded, flat)


def test_each_sample_in_a_batch_is_predicted_independently_of_its_neighbours() -> None:
    model = tiny_transformer()
    latents, text, timestep = tiny_inputs(batch=2)

    batched = model(latents, text, timestep, latent_h=2, latent_w=3)
    alone = model(latents[1:], text[1:], timestep[1:], latent_h=2, latent_w=3)

    torch.testing.assert_close(batched[1:], alone, atol=1e-5, rtol=1e-4)


def test_the_prediction_depends_on_the_text_the_timestep_and_the_image() -> None:
    model = tiny_transformer()
    latents, text, timestep = tiny_inputs(batch=1)
    base = model(latents, text, timestep, latent_h=2, latent_w=3)

    other_text = model(latents, text + 1.0, timestep, latent_h=2, latent_w=3)
    other_time = model(latents, text, timestep * 0.5, latent_h=2, latent_w=3)
    other_image = model(latents + 1.0, text, timestep, latent_h=2, latent_w=3)

    for changed in (other_text, other_time, other_image):
        assert not torch.allclose(changed, base)


def test_text_tokens_attend_causally_and_never_to_the_image_while_the_image_sees_everything() -> None:
    model = tiny_transformer()
    calls = _capture_block_calls(model)

    _forward(model, batch=1, h=1, w=2, text_len=2)

    mask = calls[0][1]["attn_mask"]
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_only_image_tokens_get_the_real_timestep_and_the_text_gets_the_zero_timestep_row() -> None:
    model = tiny_transformer()
    calls = _capture_block_calls(model)

    _forward(model, batch=2, h=1, w=2, text_len=3)

    args, kwargs = calls[0]
    assert args[1].shape == (3, 4 * TINY_CONFIG["dim"])
    assert kwargs["target_token_mask"].tolist() == [False] * 3 + [True] * 2
    zero_row = model.modulation(model.time_text_embed(torch.zeros(1), torch.zeros(1, 1, 1)))[0]
    torch.testing.assert_close(args[1][-1], zero_row)


def test_the_blocks_receive_the_text_then_image_rope_frequencies() -> None:
    model = tiny_transformer()
    calls = _capture_block_calls(model)

    _forward(model, batch=1, h=2, w=3, text_len=3)

    expected = build_rope_frequencies(3, 2, 3, axes_dim=TINY_CONFIG["axes_dims_rope"])
    assert torch.equal(calls[0][1]["rotary_emb"], expected)


def test_every_parameter_is_trainable_by_default_and_receives_a_gradient() -> None:
    model = tiny_transformer()

    _forward(model).square().mean().backward()

    assert all(p.requires_grad for p in model.parameters())
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_the_state_dict_names_the_expected_top_level_modules() -> None:
    model = tiny_transformer(num_layers=1)

    roots = {key.split(".")[0] for key in model.state_dict()}

    assert roots == {"time_text_embed", "txt_in", "img_in", "modulation", "transformer_blocks", "norm_out", "proj_out"}
    assert len(model.transformer_blocks) == 1
