from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.vae import QwenImage21VAE, load_qwen_image_2_1_vae

from ._vae_helpers import randomize, use_tiny_networks

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def tiny_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    use_tiny_networks(monkeypatch)


def _source() -> QwenImage21VAE:
    return randomize(QwenImage21VAE(include_decoder=True), seed=3, std=0.1)


def _checkpoint(tmp_path: Path, drop: tuple[str, ...] = ()) -> Path:
    vae = _source()
    tensors: dict[str, torch.Tensor] = {}
    for key, value in vae.state_dict().items():
        group, _, rest = key.partition(".")
        root = {"quant_conv": "conv1.", "post_quant_conv": "conv2."}.get(group)
        tensors[f"{root}{rest}" if root else key] = value.clone()
    path = tmp_path / "vae.safetensors"
    save_file({k: v for k, v in tensors.items() if not k.startswith(drop)}, str(path))
    return path


def test_a_missing_checkpoint_is_reported_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="VAE checkpoint not found"):
        load_qwen_image_2_1_vae(tmp_path / "absent.safetensors", device=CPU)


def test_the_loader_reads_root_conv_weights_into_quant_and_post_quant_conv(tmp_path: Path) -> None:
    source = _source()

    vae = load_qwen_image_2_1_vae(_checkpoint(tmp_path), device=CPU, dtype=torch.float32, include_decoder=True)

    for name in ("quant_conv", "post_quant_conv"):
        torch.testing.assert_close(getattr(vae, name).weight, getattr(source, name).weight)
    torch.testing.assert_close(vae.encoder.conv1.weight, source.encoder.conv1.weight)
    torch.testing.assert_close(vae.decoder.conv1.weight, source.decoder.conv1.weight)


def test_without_the_decoder_only_the_encoder_side_is_loaded(tmp_path: Path) -> None:
    vae = load_qwen_image_2_1_vae(str(_checkpoint(tmp_path)), device=CPU, dtype=torch.float32)

    assert vae.decoder is None
    assert vae.post_quant_conv is None
    assert vae.encode(torch.rand(1, 3, 16, 16)).shape == (1, 64, 1, 1)


def test_the_loaded_vae_is_frozen_in_eval_mode_and_cast_to_the_dtype(tmp_path: Path) -> None:
    vae = load_qwen_image_2_1_vae(_checkpoint(tmp_path), device=CPU, dtype=torch.bfloat16, include_decoder=True)

    assert not any(p.requires_grad for p in vae.parameters())
    assert not vae.training
    assert {p.dtype for p in vae.parameters()} == {torch.bfloat16}


def test_a_checkpoint_without_the_root_quant_conv_is_rejected(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, drop=("conv1.",))

    with pytest.raises(KeyError, match=r"no root `conv1\.\*`"):
        load_qwen_image_2_1_vae(path, device=CPU)


def test_a_checkpoint_without_the_root_post_quant_conv_is_rejected_when_decoding(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, drop=("conv2.",))

    with pytest.raises(KeyError, match=r"no root `conv2\.\*`"):
        load_qwen_image_2_1_vae(path, device=CPU, include_decoder=True)


def test_the_missing_post_quant_conv_does_not_matter_when_the_decoder_is_not_requested(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, drop=("conv2.", "decoder."))

    assert load_qwen_image_2_1_vae(path, device=CPU).decoder is None


def test_an_encoder_weight_missing_from_the_checkpoint_fails_the_strict_load(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, drop=("encoder.head.2.bias",))

    with pytest.raises(RuntimeError, match=r"encoder\.head\.2\.bias|head\.2\.bias"):
        load_qwen_image_2_1_vae(path, device=CPU)
