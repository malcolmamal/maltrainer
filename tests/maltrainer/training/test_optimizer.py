from __future__ import annotations

import pytest
import torch

from maltrainer.optimizers import Automagic3
from maltrainer.training.optimizer import build_lora_optimizer, warmup_scheduler


class _Adapters:
    """What the builder asks of a LoRA network: its parameters, and the same split into named groups."""

    def __init__(self) -> None:
        self.attn, self.mlp = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [self.attn, self.mlp]

    def param_groups(self, split: bool) -> list[dict[str, object]]:
        if not split:
            return [{"params": [self.attn, self.mlp], "name": "lora"}]
        return [{"params": [self.attn], "name": "attn"}, {"params": [self.mlp], "name": "mlp"}]


def test_automagic3_gets_the_network_s_groups_and_the_trainer_s_sign_window() -> None:
    optimizer, resolved = build_lora_optimizer(_Adapters(), {"learning_rate": 2e-6}, 8, strict=True)

    assert isinstance(optimizer, Automagic3)
    assert resolved["groups"] == ["attn", "mlp"]
    assert (resolved["lr"], resolved["polarity_history"]) == (2e-6, 8)


def test_an_unknown_optimizer_is_an_error_when_strict() -> None:
    with pytest.raises(ValueError, match=r"Unknown train.optimizer 'sgd': use automagic3, prodigy"):
        build_lora_optimizer(_Adapters(), {"optimizer": "SGD"}, 16, strict=True)


@pytest.mark.parametrize(("name", "strict"), [("adamw", True), ("sgd", False)])
def test_adamw_is_named_or_the_fallback(name: str, strict: bool) -> None:
    optimizer, resolved = build_lora_optimizer(_Adapters(), {"optimizer": name, "weight_decay": 0.1}, 16, strict)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert resolved == {"name": "adamw", "lr": 1e-6}
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_warmup_ramps_the_rate_over_its_steps_and_is_absent_without_it() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)

    assert warmup_scheduler(optimizer, {"learning_rate_warmup_steps": 0}) is None
    scheduler = warmup_scheduler(optimizer, {"learning_rate_warmup_steps": 4})
    rates = []
    for _ in range(5):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == [0.25, 0.5, 0.75, 1.0, 1.0]


def test_automagic3_gets_no_warmup_because_it_would_only_lower_its_start_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Automagic3 reads the group rate once, when a parameter's state is created, then adapts it by itself:
    # a scheduler would not ramp it up, it would leave every parameter starting at lr / steps for good.
    optimizer, _ = build_lora_optimizer(_Adapters(), {"learning_rate": 1e-4}, 8, strict=True)

    with caplog.at_level("WARNING"):
        assert warmup_scheduler(optimizer, {"learning_rate_warmup_steps": 10}) is None

    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert "Automagic3" in caplog.text
