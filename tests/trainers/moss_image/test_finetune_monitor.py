"""The [monitor] health line, the weight-drift tracker and the stop on a NaN loss."""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest
import torch

import train_moss_image_finetune as finetune
from maltrainer.utils.train_monitor import NonFiniteLossError

from .runs import run_finetune
from .world import Workspace

SIGNALS = {"automagic_polarity_history": 2, "monitor_every": 1}


def _monitor_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("[monitor]")]


def test_each_summary_gives_the_loss_bands_prior_signal_drift_and_nonfinite_counts(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2, caption="a photo of zed")

    with caplog.at_level(logging.INFO):
        run_finetune(workspace.config(prior={"weight": 1.0, "samples_per_image": 1}, train={**SIGNALS, "max_steps": 3}))

    last = _monitor_lines(caplog)[-1]
    assert last.startswith("[monitor] step 3/3 | loss ")
    assert re.search(r"\((low|mid|high) [\d.]+ n=\d+", last) and re.search(r"\| prior [\d.]+ \|", last)
    assert re.search(r"signal attn [+-][\d.]+ mlp [+-][\d.]+", last) and "nonfinite 0" in last
    assert re.search(r"drift attn [\d.e+-]+ mlp [\d.e+-]+ \(t\^[\d.-]+\)", last) and "s/step" in last


def test_drift_is_tracked_for_the_attention_output_and_feed_forward_down_weights_of_the_first_block(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)

    with caplog.at_level(logging.INFO):
        run_finetune(workspace.config(train={"max_steps": 1}))

    assert "Monitor: weight drift of 2 tensors" in caplog.text


@pytest.mark.parametrize(
    "sections",
    [{"train": {"monitor_drift": False}}, {"finetune": {"trainable_prefixes": ["layers.1."]}}],
    ids=["switched_off", "no_tracked_block_trains"],
)
def test_no_drift_is_tracked_when_it_is_off_or_none_of_its_blocks_train(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, sections: dict[str, Any]
) -> None:
    workspace.add_images("p", 1)

    with caplog.at_level(logging.INFO):
        run_finetune(workspace.config(sections, train={"max_steps": 1, "monitor_every": 1}))

    assert "Monitor: weight drift" not in caplog.text and "drift " not in _monitor_lines(caplog)[-1]


def test_three_nan_losses_in_a_row_stop_the_run_without_saving_anything(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)

    def nan_step(self: finetune.MOSSImageFinetuneTrainer, _index: int) -> torch.Tensor:
        self.last_sigma = torch.tensor(0.5)
        return torch.full((), float("nan"), requires_grad=True)

    monkeypatch.setattr(finetune.MOSSImageFinetuneTrainer, "_training_step", nan_step)

    with pytest.raises(NonFiniteLossError, match=r"the loss was NaN / inf 3 steps in a row; stopping without saving"):
        run_finetune(workspace.config(train={"max_steps": 6}))

    assert not (workspace.out / "ft.safetensors").exists() and not (workspace.out / "ft_state").exists()
    assert "An error occurred during training" in caplog.text
