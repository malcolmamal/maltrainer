"""What a resume keeps of the sign history: the valid ranges of its counters and its width per parameter size."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3

HISTORY = 4


@pytest.mark.parametrize(
    ("numel", "hist_idx", "hist_fill"),
    [
        (5, 2, 3),
        (8, 2, 3),  # 8 values pack into 1 byte: (n + 7) // 8, not (n + 8) // 8
        (9, 2, 3),  # 9 values need 2 bytes: (n + 7) // 8, not // 9
        (5, 0, 3),  # an index of 0 is the start of the ring, valid
        (5, 2, 0),  # a fill of 0 is an empty window, valid
        (5, 3, HISTORY),  # a full window and the last index are valid
    ],
)
def test_a_valid_sign_history_is_kept_as_it_was_saved(numel: int, hist_idx: int, hist_fill: int) -> None:
    p = nn.Parameter(torch.zeros(numel))
    opt = Automagic3([p], lr=1e-4, polarity_history=HISTORY)
    p.grad = torch.ones(numel)
    opt.step()
    width = (numel + 7) // 8
    state = opt.state_dict()
    saved = torch.full((HISTORY, width), 0xA5, dtype=torch.uint8)
    state["state"][0].update(sign_history=saved.clone(), hist_idx=hist_idx, hist_fill=hist_fill)

    clone = nn.Parameter(torch.zeros(numel))
    resumed = Automagic3([clone], lr=1e-4, polarity_history=HISTORY)
    resumed.load_state_dict(state)

    st = resumed.state[clone]
    torch.testing.assert_close(st["sign_history"], saved, rtol=0, atol=0)
    assert (st["hist_idx"], st["hist_fill"]) == (hist_idx, hist_fill)
