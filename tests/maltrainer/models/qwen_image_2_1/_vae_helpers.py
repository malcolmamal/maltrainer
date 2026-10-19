from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from maltrainer.models.qwen_image_2_1 import vae as vae_module
from maltrainer.models.qwen_image_2_1.vae import CausalConv3d, QwenImage21VAE


def randomize(module: nn.Module, seed: int = 0, std: float = 0.2) -> nn.Module:
    """The VAE convolutions are created with uninitialised memory, so tests fill every parameter."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return module


def make_identity(conv: CausalConv3d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        for channel in range(conv.weight.shape[0]):
            conv.weight[channel, channel, 0, 0, 0] = 1.0


class StubEncoder(nn.Module):
    """Returns whatever moments a test installs and remembers the pixels it was shown."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.moments = torch.zeros(1, 2 * z_dim, 1, 1)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x)
        return self.moments.to(x.dtype).expand(x.shape[0], -1, -1, -1)


class StubDecoder(nn.Module):
    """Hands the first four latent channels back as the RGBA image."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.seen.append(z)
        return z[:, :4]


class TinyDecoder(nn.Module):
    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = CausalConv3d(z_dim, 4, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv1(z)


def use_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vae_module, "Qwen21VAEEncoder", StubEncoder)
    monkeypatch.setattr(vae_module, "Qwen21VAEDecoder", StubDecoder)


def use_tiny_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vae_module, "Qwen21VAEEncoder", partial(vae_module.Qwen21VAEEncoder, base_dim=4))
    monkeypatch.setattr(vae_module, "Qwen21VAEDecoder", TinyDecoder)


def stub_vae(monkeypatch: pytest.MonkeyPatch, include_decoder: bool = True) -> QwenImage21VAE:
    use_stubs(monkeypatch)
    vae = QwenImage21VAE(include_decoder=include_decoder)
    make_identity(vae.quant_conv)
    if include_decoder:
        make_identity(vae.post_quant_conv)
    return vae
