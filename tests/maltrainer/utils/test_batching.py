from __future__ import annotations

import pytest
import torch

from maltrainer.utils.batching import batch_text_embeddings


def test_equally_long_prompts_become_one_batch() -> None:
    first, second = torch.ones(1, 4, 8), torch.zeros(1, 4, 8)

    batch = batch_text_embeddings([first, second])

    assert batch.shape == (2, 4, 8)
    assert torch.equal(batch[0], first[0]) and torch.equal(batch[1], second[0])


def test_a_single_prompt_is_a_batch_of_one() -> None:
    assert batch_text_embeddings([torch.ones(1, 5, 8)]).shape == (1, 5, 8)


def test_prompts_of_different_lengths_are_refused_with_the_way_out() -> None:
    with pytest.raises(ValueError, match=r"different token lengths.*batch_size 1.*gradient_accumulation_steps"):
        batch_text_embeddings([torch.ones(1, 4, 8), torch.ones(1, 6, 8)])
