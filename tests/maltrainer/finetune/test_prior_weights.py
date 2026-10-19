from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.finetune.prior import weights_from_file


def _linear() -> nn.Linear:
    linear = nn.Linear(3, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(6.0).reshape(2, 3))
        linear.bias.copy_(torch.tensor([-1.0, -2.0]))
    return linear


def _file(tmp_path: Path, tensors: dict) -> Path:
    path = tmp_path / "base.safetensors"
    save_file(tensors, str(path))
    return path


def test_the_file_values_hold_inside_and_the_trained_ones_return_after(tmp_path: Path) -> None:
    linear = _linear()
    trained_weight, trained_bias = linear.weight.detach().clone(), linear.bias.detach().clone()
    path = _file(
        tmp_path,
        {"diffusion_model.weight": torch.ones(2, 3, dtype=torch.float64), "diffusion_model.bias": torch.zeros(2)},
    )

    with weights_from_file(list(linear.named_parameters()), path):
        inside = linear.weight.detach().clone(), linear.bias.detach().clone()

    torch.testing.assert_close(inside[0], torch.ones(2, 3))
    torch.testing.assert_close(inside[1], torch.zeros(2))
    assert inside[0].dtype == torch.float32
    assert torch.equal(linear.weight, trained_weight)
    assert torch.equal(linear.bias, trained_bias)


def test_an_unprefixed_key_wins_over_a_container_prefixed_one(tmp_path: Path) -> None:
    linear = _linear()
    path = _file(tmp_path, {"weight": torch.ones(2, 3), "model.diffusion_model.weight": torch.zeros(2, 3)})

    with weights_from_file([("weight", linear.weight)], path):
        inside = linear.weight.detach().clone()

    torch.testing.assert_close(inside, torch.ones(2, 3))


def test_custom_prefixes_choose_where_to_look(tmp_path: Path) -> None:
    linear = _linear()
    path = _file(tmp_path, {"weight": torch.ones(2, 3), "unet.weight": torch.full((2, 3), 7.0)})

    with weights_from_file([("weight", linear.weight)], path, prefixes=("unet.",)):
        inside = linear.weight.detach().clone()

    torch.testing.assert_close(inside, torch.full((2, 3), 7.0))


def test_a_parameter_missing_from_the_file_is_an_error_and_nothing_stays_changed(tmp_path: Path) -> None:
    linear = _linear()
    trained = linear.weight.detach().clone()
    path = _file(tmp_path, {"weight": torch.ones(2, 3)})

    with (
        pytest.raises(KeyError, match="base.safetensors has no tensor for bias"),
        weights_from_file(list(linear.named_parameters()), path),
    ):
        pass

    assert torch.equal(linear.weight, trained)


def test_the_trained_values_return_when_the_body_raises(tmp_path: Path) -> None:
    linear = _linear()
    trained = linear.weight.detach().clone()
    path = _file(tmp_path, {"weight": torch.ones(2, 3)})

    with pytest.raises(RuntimeError, match="recording failed"), weights_from_file([("weight", linear.weight)], path):
        raise RuntimeError("recording failed")

    assert torch.equal(linear.weight, trained)
