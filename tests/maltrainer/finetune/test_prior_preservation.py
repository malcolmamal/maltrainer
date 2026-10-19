from __future__ import annotations

import random
from pathlib import Path
from typing import List

import pytest
import torch

from maltrainer.finetune.prior import PriorPreservation

X0 = torch.tensor([[[0.5, -0.5], [1.0, 0.0]]])
CPU = torch.device("cpu")


class Recorder:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: List[tuple] = []

    def __call__(self, x_t: torch.Tensor, sigma: torch.Tensor, caption: str) -> torch.Tensor:
        self.calls.append((x_t.clone(), sigma.clone(), caption))
        return 2.0 * x_t + self.offset


def _sigmas(*values: float):
    queue = list(values)
    return lambda: torch.tensor(queue.pop(0))


def _built(samples: int = 2) -> PriorPreservation:
    prior = PriorPreservation(weight=1.0, samples_per_image=samples, seed=5)
    prior.build([("a", X0, "photo of woman")], _sigmas(0.25, 0.75), Recorder(), CPU)
    return prior


def test_settings_are_clamped_to_at_least_one() -> None:
    prior = PriorPreservation(weight=0, samples_per_image=0, every=-2, seed=3)

    assert (prior.weight, prior.samples_per_image, prior.every, prior.seed) == (0.0, 1, 1, 3)


def test_noising_blends_the_image_toward_noise_by_sigma() -> None:
    x0, noise = torch.ones(2, 3), torch.zeros(2, 3)

    out = PriorPreservation.noisy(x0, noise, torch.tensor([0.25, 1.0]))

    torch.testing.assert_close(out, torch.tensor([[0.75] * 3, [0.0] * 3]))


@pytest.mark.parametrize(
    ("weight", "every", "steps", "expected"),
    [
        (1.0, 1, [0, 1, 2], [True, True, True]),
        (1.0, 3, [0, 1, 3, 4], [True, False, True, False]),
        (0.0, 1, [0], [False]),
    ],
)
def test_the_prior_term_is_due_every_nth_step_while_it_has_weight(
    weight: float, every: int, steps: List[int], expected: List[bool]
) -> None:
    prior = PriorPreservation(weight=weight, every=every)

    assert [prior.due(step) for step in steps] == expected


def test_build_records_each_inputs_sigma_seed_and_prediction() -> None:
    predict = Recorder()
    prior = PriorPreservation(samples_per_image=2, seed=5)

    prior.build([("a", X0, "photo of woman")], _sigmas(0.25, 0.75), predict, CPU)

    assert ("a" in prior, "b" in prior, len(prior), prior.keys()) == (True, False, 1, ["a"])
    assert prior.captions == {"a": "photo of woman"}
    torch.testing.assert_close(prior.sigmas["a"], torch.tensor([0.25, 0.75]))
    rng = random.Random(5)
    assert prior.seeds["a"].tolist() == [rng.getrandbits(62), rng.getrandbits(62)]
    assert [(tuple(x.shape), s.tolist(), c) for x, s, c in predict.calls] == [
        ((1, 1, 2, 2), [0.25], "photo of woman"),
        ((1, 1, 2, 2), [0.75], "photo of woman"),
    ]
    expected = torch.stack([2.0 * x[0] for x, _, _ in predict.calls]).to(torch.bfloat16)
    assert prior.targets["a"].dtype == torch.bfloat16
    torch.testing.assert_close(prior.targets["a"], expected)


def test_the_same_seed_records_the_same_noise() -> None:
    first, second = _built(), _built()

    assert torch.equal(first.seeds["a"], second.seeds["a"])
    assert torch.equal(first.targets["a"], second.targets["a"])


def test_the_loss_replays_a_recorded_input_exactly() -> None:
    prior = _built(samples=1)
    replay = Recorder()

    loss = prior.loss("a", X0.unsqueeze(0), replay)

    assert loss is not None
    assert replay.calls[0][2] == "photo of woman"
    torch.testing.assert_close(replay.calls[0][1], torch.tensor([0.25]))
    torch.testing.assert_close(2.0 * replay.calls[0][0][0], prior.targets["a"][0].float(), atol=0.02, rtol=0.0)
    assert float(loss) < 1e-4


def test_the_loss_measures_how_far_the_model_moved() -> None:
    prior = _built()
    random.seed(0)

    loss = prior.loss("a", X0.unsqueeze(0), Recorder(offset=0.5))

    assert float(loss) == pytest.approx(0.25, abs=1e-3)


def test_an_image_without_recorded_inputs_has_no_loss() -> None:
    assert _built().loss("b", X0.unsqueeze(0), Recorder()) is None


def test_a_saved_prior_loads_back_into_a_fresh_one(tmp_path: Path) -> None:
    prior = _built()
    prior.build([("0-first", X0 * 2, "photo of man")], _sigmas(0.5, 0.5), Recorder(), CPU)
    path = tmp_path / "prior.safetensors"

    prior.save(path)
    loaded = PriorPreservation()
    loaded.load(path)

    assert [p.name for p in tmp_path.iterdir()] == ["prior.safetensors"]
    assert sorted(loaded.keys()) == ["0-first", "a"]
    assert loaded.captions == prior.captions
    for key in prior.keys():
        assert torch.equal(loaded.targets[key], prior.targets[key])
        assert torch.equal(loaded.sigmas[key], prior.sigmas[key])
        assert torch.equal(loaded.seeds[key], prior.seeds[key])
        assert loaded.seeds[key].dtype == torch.int64


def test_loading_a_prior_recorded_with_another_sample_count_warns_and_keeps_the_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "prior.safetensors"
    _built(samples=2).save(path)
    loaded = PriorPreservation(samples_per_image=4)

    with caplog.at_level("WARNING", logger="maltrainer.finetune.prior"):
        loaded.load(path)

    assert loaded.samples_per_image == 4
    assert "has 2 inputs per image; new images get 4" in caplog.text
    assert loaded.targets["a"].shape[0] == 2


def test_loading_a_prior_recorded_with_the_same_sample_count_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "prior.safetensors"
    _built(samples=2).save(path)

    with caplog.at_level("WARNING", logger="maltrainer.finetune.prior"):
        PriorPreservation(samples_per_image=2).load(path)

    assert caplog.text == ""
