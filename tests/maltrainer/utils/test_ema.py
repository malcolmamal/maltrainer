from __future__ import annotations

import pytest
import torch

from maltrainer.utils.ema import OffloadedParameterEMA, ParameterEMA


def _param(*values: float, dtype: torch.dtype = torch.float32, trainable: bool = True) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(values, dtype=dtype), requires_grad=trainable)


def _set(param: torch.nn.Parameter, *values: float) -> None:
    with torch.no_grad():
        param.copy_(torch.tensor(values, dtype=param.dtype))


def test_only_trainable_parameters_are_averaged() -> None:
    trained, frozen = _param(1.0), _param(2.0, trainable=False)

    ema = ParameterEMA([trained, frozen])

    assert len(ema.parameters) == 1
    assert ema.parameters[0] is trained
    torch.testing.assert_close(ema.shadow[0], torch.tensor([1.0]))


def test_the_warmup_ramps_the_decay_up_to_its_ceiling() -> None:
    ema = ParameterEMA([_param(0.0)], decay=0.3)
    seen = []

    for _ in range(4):
        seen.append(ema.current_decay())
        ema.update()

    assert seen == pytest.approx([0.1, 2 / 11, 0.25, 0.3])


def test_without_warmup_the_decay_is_constant() -> None:
    ema = ParameterEMA([_param(0.0)], decay=0.98, warmup=False)

    ema.update()

    assert ema.current_decay() == 0.98
    assert ema.updates == 1


def test_an_update_moves_the_shadow_towards_the_live_weights() -> None:
    param = _param(0.0, 0.0)
    ema = ParameterEMA([param])
    _set(param, 10.0, 20.0)

    ema.update()

    torch.testing.assert_close(ema.shadow[0], torch.tensor([9.0, 18.0]))


def test_the_average_is_applied_in_the_parameter_dtype_and_then_restored() -> None:
    param = _param(1.0, dtype=torch.bfloat16)
    ema = ParameterEMA([param], decay=0.5, warmup=False)
    _set(param, 2.0)
    ema.update()

    with ema.applied():
        inside = param.detach().clone()

    assert ema.shadow[0].dtype == torch.float32
    torch.testing.assert_close(inside, torch.tensor([1.5], dtype=torch.bfloat16))
    torch.testing.assert_close(param.detach(), torch.tensor([2.0], dtype=torch.bfloat16))


def test_the_raw_weights_come_back_when_the_applied_block_raises() -> None:
    param = _param(4.0)
    ema = ParameterEMA([param])
    _set(param, 8.0)

    with pytest.raises(RuntimeError, match="render failed"), ema.applied():
        raise RuntimeError("render failed")

    torch.testing.assert_close(param.detach(), torch.tensor([8.0]))


def test_the_offloaded_shadow_names_only_trainable_parameters() -> None:
    named = [("a", _param(1.0, dtype=torch.bfloat16)), ("b", _param(2.0, trainable=False))]

    ema = OffloadedParameterEMA(named)

    assert ema.names == ["a"]
    assert ema.averaged().keys() == {"a"}


@pytest.mark.parametrize(("warmup", "every", "expected"), [(True, 1, 0.1), (True, 2, 0.01), (False, 3, 0.125)])
def test_the_offloaded_decay_is_raised_to_the_update_interval(warmup: bool, every: int, expected: float) -> None:
    ema = OffloadedParameterEMA([("w", _param(0.0, dtype=torch.bfloat16))], decay=0.5, warmup=warmup, every=every)

    assert ema.current_decay() == pytest.approx(expected)


def test_an_offloaded_update_averages_on_the_cpu_in_fp32() -> None:
    param = _param(0.0, 0.0, dtype=torch.bfloat16)
    ema = OffloadedParameterEMA([("w", param)], decay=0.5, warmup=False)
    _set(param, 4.0, 8.0)

    ema.update()

    averaged = ema.averaged()
    torch.testing.assert_close(averaged["w"], torch.tensor([2.0, 4.0]))
    assert averaged["w"].dtype == torch.float32


def test_offloaded_updates_between_intervals_only_count_the_call() -> None:
    param = _param(0.0, dtype=torch.bfloat16)
    ema = OffloadedParameterEMA([("w", param)], decay=0.5, warmup=False, every=2)
    _set(param, 4.0)

    ema.update()
    skipped = ema.averaged()["w"].clone()
    ema.update()

    torch.testing.assert_close(skipped, torch.tensor([0.0]))
    torch.testing.assert_close(ema.averaged()["w"], torch.tensor([3.0]))
    assert ema.state()[1] == {"updates": 1, "calls": 2}


def test_the_offloaded_average_is_applied_and_the_raw_weights_restored_bit_for_bit() -> None:
    param = _param(0.0, dtype=torch.bfloat16)
    ema = OffloadedParameterEMA([("w", param)], decay=0.5, warmup=False)
    _set(param, 3.0)
    ema.update()

    with pytest.raises(RuntimeError, match="render failed"), ema.applied():
        inside = param.detach().clone()
        raise RuntimeError("render failed")

    torch.testing.assert_close(inside, torch.tensor([1.5], dtype=torch.bfloat16))
    torch.testing.assert_close(param.detach(), torch.tensor([3.0], dtype=torch.bfloat16))


def test_a_saved_offloaded_state_loads_into_a_fresh_average() -> None:
    first = OffloadedParameterEMA([("w", _param(1.0, dtype=torch.bfloat16))], every=2)
    first.shadow[0].fill_(0.25)
    first.updates, first.calls = 3, 7
    tensors, counters = first.state()
    fresh = OffloadedParameterEMA([("w", _param(9.0, dtype=torch.bfloat16))], every=2)

    fresh.load_state(tensors, counters)

    torch.testing.assert_close(fresh.averaged()["w"], torch.tensor([0.25]))
    assert (fresh.updates, fresh.calls) == (3, 7)


def test_loading_a_state_without_counters_starts_them_at_zero() -> None:
    ema = OffloadedParameterEMA([("w", _param(1.0, dtype=torch.bfloat16))])
    ema.updates, ema.calls = 5, 5

    ema.load_state({"w": torch.tensor([0.5])}, {})

    assert (ema.updates, ema.calls) == (0, 0)


def test_loading_a_state_that_misses_a_trained_parameter_fails() -> None:
    named = [("a", _param(1.0, dtype=torch.bfloat16)), ("b", _param(2.0, dtype=torch.bfloat16))]
    ema = OffloadedParameterEMA(named)

    with pytest.raises(ValueError, match=r"lacks 1 trained parameters, e.g. \['b'\]"):
        ema.load_state({"a": torch.tensor([0.0])}, {})
