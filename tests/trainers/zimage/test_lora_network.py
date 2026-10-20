"""How the config shapes the adapters: rank, alpha, targeted modules, key prefix and dtype."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from .runs import prepared_lora, run_lora
from .world import Workspace, lora_layers


def _saved(ws: Workspace) -> dict[str, torch.Tensor]:
    return load_file(str(ws.out / "zimage_lora_step_final.safetensors"))


def test_rank_and_alpha_size_the_down_and_up_projections_and_scale_the_output(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared_lora(workspace.lora_config(lora={"rank": 2, "alpha": 6}))

    layer = lora_layers(trainer)[0]
    assert (layer.lora_down.weight.shape, layer.lora_up.weight.shape) == ((2, 16), (16, 2))
    assert (layer.alpha, layer.scale) == (6, 3.0)
    assert trainer.lora_network.prefix == ""


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (["to_k"], {f"{block}.attention.to_k" for block in ("noise_refiner.0", "context_refiner.0", "layers.0")}),
        (["layers.0.attention.to_out"], {"layers.0.attention.to_out.0"}),
        (
            ["w2", "cap_embedder"],
            {f"{b}.feed_forward.w2" for b in ("noise_refiner.0", "context_refiner.0", "layers.0")} | {"cap_embedder.1"},
        ),
        (["no_such_module"], set()),
    ],
)
def test_target_modules_picks_the_linears_whose_name_contains_one_of_them(
    workspace: Workspace, targets: list[str], expected: set[str]
) -> None:
    workspace.add_images(1)

    trainer = prepared_lora(workspace.lora_config(lora={"target_modules": targets}))

    assert set(trainer.lora_network.name_map.values()) == expected


def test_the_prefix_leads_every_saved_key_and_the_alpha_is_a_scalar_tensor(workspace: Workspace) -> None:
    workspace.add_images(1)

    run_lora(workspace.lora_config(lora={"prefix": "diffusion_model."}))

    saved = _saved(workspace)
    assert {key.split(".")[0] for key in saved} == {"diffusion_model"}
    assert saved["diffusion_model.layers.0.attention.to_q.alpha"].shape == ()


def test_a_bfloat16_run_trains_and_saves_the_adapters_in_bfloat16(workspace: Workspace) -> None:
    workspace.add_images(2)
    config: dict[str, Any] = workspace.lora_config(dtype="bfloat16")

    trainer = run_lora(config)

    assert {t.dtype for t in _saved(workspace).values() if t.ndim} == {torch.bfloat16}
    assert next(trainer.transformer.parameters()).dtype == torch.bfloat16
    assert all(torch.count_nonzero(layer.lora_up.weight) > 0 for layer in lora_layers(trainer))
