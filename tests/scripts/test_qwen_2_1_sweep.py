"""Pure parts of the Qwen-Image-2.1 sampling scripts: sigma schedule, checkpoint naming, the Euler loop, adapter swap.

The scripts' ``main`` functions need a GPU and the real 2.1 weights (they build ``torch.device("cuda")``), so
they are not run here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

import qwen_2_1_bench_generate as bench_generate
import qwen_2_1_sample
import qwen_2_1_sweep_checkpoints as sweep
from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork

BOTH = pytest.mark.parametrize("module", [qwen_2_1_sample, sweep])
CPU = torch.device("cpu")


@BOTH
def test_the_schedule_runs_from_one_down_to_the_terminal_sigma_and_ends_at_zero(module: Any) -> None:
    sigmas = module.build_sigmas(10, 1024, CPU)

    assert sigmas.shape == (11,) and sigmas.dtype == torch.float32
    assert (
        sigmas[0].item() == pytest.approx(1.0) and sigmas[-2].item() == pytest.approx(0.02) and sigmas[-1].item() == 0.0
    )
    assert bool((sigmas[:-1] > sigmas[1:]).all())


@BOTH
def test_a_larger_image_shifts_the_middle_of_the_schedule_towards_noise(module: Any) -> None:
    small, large = module.build_sigmas(10, 64, CPU), module.build_sigmas(10, 4096, CPU)

    assert bool((large[1:-2] > small[1:-2]).all())


def test_the_two_scripts_build_the_same_schedule() -> None:
    assert torch.equal(qwen_2_1_sample.build_sigmas(7, 256, CPU), sweep.build_sigmas(7, 256, CPU))


@pytest.mark.parametrize(
    ("name", "key", "label"),
    [
        ("run_step_600.safetensors", (0, 600), "step 600"),
        ("run_step_5.safetensors", (0, 5), "step 5"),
        ("run.safetensors", (1, 0), "final"),
        ("run_step_x.safetensors", (1, 0), "final"),
    ],
)
def test_checkpoints_sort_by_step_with_the_unsuffixed_final_one_last(
    name: str, key: tuple[int, int], label: str
) -> None:
    assert (sweep.sort_key(Path(name)), sweep.label_of(Path(name))) == (key, label)
    assert bench_generate.step_of(Path(name)) == (label.removeprefix("step ") if key[0] == 0 else "final")


def test_a_run_is_named_after_its_output_folder_without_the_bench_prefix() -> None:
    assert bench_generate.run_name({"output": {"path": "output/bench_adamw"}}) == "adamw"
    assert bench_generate.run_name({"output": {"path": "output/rhea_prodigy/"}}) == "rhea_prodigy"


def test_the_default_prompts_each_carry_an_id_a_prompt_and_yes_no_checks() -> None:
    prompts = bench_generate.DEFAULT_PROMPTS

    assert len({p["id"] for p in prompts}) == len(prompts)
    assert all(p["prompt"] and p["checks"] and all(c.endswith("?") for c in p["checks"]) for p in prompts)


def test_the_euler_loop_adds_sigma_steps_times_the_models_velocity_to_noise_from_the_seed() -> None:
    seen: list[dict[str, Any]] = []

    def transformer(**kwargs: Any) -> torch.Tensor:
        seen.append({"t": kwargs["timestep"].item(), "hw": (kwargs["latent_h"], kwargs["latent_w"])})
        return kwargs["hidden_states"]

    sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])

    out = sweep.sample_latents(transformer, "embeds", sigmas, 3, 2, 3, torch.float32, CPU)

    noise = torch.randn((1, 64, 2, 3), generator=torch.Generator().manual_seed(3))
    assert out.shape == (1, 64, 2, 3)
    assert torch.allclose(out, noise * (0.5 * 0.75 * 0.75))
    assert seen == [{"t": 1.0, "hw": (2, 3)}, {"t": 0.5, "hw": (2, 3)}, {"t": 0.25, "hw": (2, 3)}]


def test_a_different_seed_gives_different_latents() -> None:
    def transformer(**kwargs: Any) -> torch.Tensor:
        return torch.zeros_like(kwargs["hidden_states"])

    one = sweep.sample_latents(transformer, None, torch.tensor([1.0, 0.0]), 1, 2, 2, torch.float32, CPU)
    two = sweep.sample_latents(transformer, None, torch.tensor([1.0, 0.0]), 2, 2, 2, torch.float32, CPU)

    assert not torch.equal(one, two)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(6, 5, bias=False)
        self.to_k = nn.Linear(6, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x) + self.to_k(x)


def _checkpoint(path: Path, seed: int, alpha: float | None) -> Path:
    gen = torch.Generator().manual_seed(seed)
    state = {}
    for name in ("to_q", "to_k"):
        state[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(2, 6, generator=gen)
        state[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(5, 2, generator=gen)
        if alpha is not None:
            state[f"diffusion_model.{name}.alpha"] = torch.tensor(alpha)
    save_file(state, str(path))
    return path


def _delta(path: Path, name: str, scale: float, alpha: float) -> torch.Tensor:
    state = load_file(str(path))
    return (
        state[f"diffusion_model.{name}.lora_B.weight"]
        @ state[f"diffusion_model.{name}.lora_A.weight"]
        * scale
        * alpha
        / 2
    )


def test_swapping_a_checkpoint_in_copies_its_weights_and_sets_the_scale_from_alpha_rank_and_strength(
    tmp_path: Path,
) -> None:
    transformer = Tiny()
    first, second = _checkpoint(tmp_path / "a.safetensors", 1, 4.0), _checkpoint(tmp_path / "b.safetensors", 2, None)
    network = QwenImage21LoRANetwork(transformer, target_modules=["__none__"])
    network.load_assistant_adapter(first)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(9))

    sweep.swap_adapter(network, second, 0.5)

    layer: Any = network.assistant_layers["to_q"]
    assert layer.scale == 0.5
    out = layer.lora_up(layer.lora_down(x)) * layer.scale
    assert torch.allclose(out, x @ _delta(second, "to_q", 0.5, 2.0).T, atol=1e-5)


def test_a_scale_of_zero_reproduces_the_base_model_and_the_alpha_is_read_when_present(tmp_path: Path) -> None:
    transformer = Tiny()
    first = _checkpoint(tmp_path / "a.safetensors", 1, 4.0)
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        plain = transformer(x)
    network = QwenImage21LoRANetwork(transformer, target_modules=["__none__"])
    network.load_assistant_adapter(first)

    sweep.swap_adapter(network, first, 1.0)
    assert network.assistant_layers["to_k"].scale == 2.0
    assert not torch.allclose(transformer(x), plain)
    sweep.swap_adapter(network, first, 0.0)

    assert torch.allclose(transformer(x), plain, atol=1e-6)
