from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.flux3.vae import (
    AttnBlock,
    Decoder,
    Downsample,
    Encoder,
    Flux2VAEDecoder,
    Flux2VAEEncoder,
    ResnetBlock,
    Upsample,
)

CH = 32


def _seeded(cls: type, *args: object, **kwargs: object) -> nn.Module:
    torch.manual_seed(0)
    return cls(*args, **kwargs).eval()


def _centre_tap(conv: nn.Conv2d) -> None:
    conv.weight.data.zero_()
    conv.weight.data[:, :, 1, 1] = torch.eye(conv.out_channels)
    conv.bias.data.zero_()


def test_a_resnet_block_adds_its_input_back_unless_the_channels_change() -> None:
    same, wider = _seeded(ResnetBlock, CH, CH), _seeded(ResnetBlock, CH, 2 * CH)
    same.conv2.weight.data.zero_()
    same.conv2.bias.data.zero_()
    x = torch.randn(1, CH, 4, 4)

    assert torch.equal(same(x), x)
    assert not hasattr(same, "nin_shortcut")
    assert wider(x).shape == (1, 2 * CH, 4, 4)
    assert wider.nin_shortcut.kernel_size == (1, 1)


def test_an_attention_block_with_a_closed_output_projection_is_the_identity() -> None:
    block = _seeded(AttnBlock, CH)
    block.proj_out.weight.data.zero_()
    block.proj_out.bias.data.zero_()
    x = torch.randn(2, CH, 3, 3)

    assert torch.equal(block(x), x)


def test_an_attention_block_mixes_every_pixel_into_every_other() -> None:
    block = _seeded(AttnBlock, CH)
    x = torch.randn(1, CH, 3, 3)
    changed = x.clone()
    changed[..., 0, 0] += 5.0

    delta = block(changed) - block(x)

    assert (delta[..., 2, 2].abs() > 1e-6).any()


def test_upsampling_repeats_each_pixel_into_a_two_by_two_block_before_the_convolution() -> None:
    up = Upsample(1)
    _centre_tap(up.conv)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    out = up(x)

    assert out.shape == (1, 1, 4, 4)
    assert out[0, 0].tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


def test_downsampling_pads_right_and_bottom_and_keeps_every_second_pixel() -> None:
    down = Downsample(1)
    down.conv.weight.data.zero_()
    down.conv.weight.data[0, 0, 0, 0] = 1.0
    down.conv.bias.data.zero_()
    x = torch.arange(16.0).view(1, 1, 4, 4)

    out = down(x)

    assert out[0, 0].tolist() == [[0, 2], [8, 10]]


def test_the_encoder_downsamples_eight_times_to_twice_the_latent_channels() -> None:
    encoder = _seeded(Encoder, ch=CH, z_channels=3)

    assert encoder(torch.randn(1, 3, 32, 48)).shape == (1, 6, 4, 6)


def test_the_decoder_upsamples_eight_times_to_rgb() -> None:
    decoder = _seeded(Decoder, ch=CH, z_channels=3)

    assert decoder(torch.randn(1, 3, 4, 6)).shape == (1, 3, 32, 48)


def test_the_two_models_hold_the_batch_norm_statistics_of_the_128_channel_latent_as_buffers() -> None:
    encoder, decoder = Flux2VAEEncoder(ch=CH, embed_dim=32), Flux2VAEDecoder(ch=CH, embed_dim=32)

    for model in (encoder, decoder):
        assert model.bn_running_mean.shape == model.bn_running_var.shape == (128,)
        assert {"bn_running_mean", "bn_running_var"} <= set(model.state_dict())


def test_encoding_and_decoding_a_batch_keeps_the_sixteenfold_geometry() -> None:
    encoder = _seeded(Flux2VAEEncoder, ch=CH, z_channels=4, embed_dim=2)
    decoder = _seeded(Flux2VAEDecoder, ch=CH, z_channels=4, embed_dim=2)
    images = torch.rand(2, 3, 32, 48)

    latents = encoder.encode(images)
    decoded = decoder.decode(latents)

    assert latents.shape == (2, 8, 2, 3) and latents.dtype == torch.float32
    assert decoded.shape == (2, 3, 32, 48) and decoded.dtype == torch.float32
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0
    assert not latents.requires_grad
