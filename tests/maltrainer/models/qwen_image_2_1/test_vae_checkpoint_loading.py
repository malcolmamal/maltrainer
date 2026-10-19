from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.vae import QwenImage21VAE, load_qwen_image_2_1_vae

from ._vae_helpers import randomize, use_tiny_networks

CPU = torch.device("cpu")


def _checkpoint(path: Path, monkeypatch: pytest.MonkeyPatch, decoder: bool = True, drop: str | None = None) -> dict:
    """Write a checkpoint with the real key layout: encoder.*, decoder.*, root conv1 (quant), conv2 (post-quant)."""
    use_tiny_networks(monkeypatch)
    source = randomize(QwenImage21VAE(include_decoder=decoder), seed=3)
    tensors = {f"encoder.{k}": v for k, v in source.encoder.state_dict().items()}
    tensors.update({f"conv1.{k}": v for k, v in source.quant_conv.state_dict().items()})
    if decoder:
        tensors.update({f"decoder.{k}": v for k, v in source.decoder.state_dict().items()})
        tensors.update({f"conv2.{k}": v for k, v in source.post_quant_conv.state_dict().items()})
    tensors = {k: v.contiguous() for k, v in tensors.items() if drop is None or not k.startswith(drop)}
    save_file(tensors, str(path))
    return tensors


def test_loading_a_missing_vae_checkpoint_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.safetensors"):
        load_qwen_image_2_1_vae(tmp_path / "nope.safetensors", device=CPU)


def test_the_encoder_only_load_routes_root_conv1_into_quant_conv_and_skips_the_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _checkpoint(tmp_path / "vae.safetensors", monkeypatch, decoder=True)

    vae = load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU, dtype=torch.float32)

    assert vae.decoder is None and vae.post_quant_conv is None
    assert torch.equal(vae.quant_conv.weight, saved["conv1.weight"])
    assert torch.equal(vae.encoder.conv1.weight, saved["encoder.conv1.weight"])


def test_the_full_load_fills_decoder_and_post_quant_conv_from_root_conv2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _checkpoint(tmp_path / "vae.safetensors", monkeypatch)

    vae = load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU, dtype=torch.float32, include_decoder=True)

    assert torch.equal(vae.post_quant_conv.weight, saved["conv2.weight"])
    assert torch.equal(vae.decoder.conv1.weight, saved["decoder.conv1.weight"])


def test_the_loaded_vae_is_frozen_in_eval_mode_and_cast_to_the_requested_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkpoint(tmp_path / "vae.safetensors", monkeypatch)

    vae = load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU, dtype=torch.bfloat16, include_decoder=True)

    assert not vae.training
    assert all(not p.requires_grad and p.dtype == torch.bfloat16 for p in vae.parameters())


def test_a_checkpoint_without_root_conv1_is_rejected_as_missing_quant_conv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkpoint(tmp_path / "vae.safetensors", monkeypatch, drop="conv1.")

    with pytest.raises(KeyError, match="quant_conv"):
        load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU)


def test_a_checkpoint_without_root_conv2_is_rejected_when_the_decoder_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkpoint(tmp_path / "vae.safetensors", monkeypatch, drop="conv2.")

    with pytest.raises(KeyError, match="post_quant_conv"):
        load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU, include_decoder=True)


def test_an_encoder_weight_missing_from_the_checkpoint_fails_the_strict_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkpoint(tmp_path / "vae.safetensors", monkeypatch, drop="encoder.head")

    with pytest.raises(RuntimeError, match="Missing key"):
        load_qwen_image_2_1_vae(tmp_path / "vae.safetensors", device=CPU)
