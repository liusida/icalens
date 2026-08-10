from __future__ import annotations

import torch

from demo.fit import parse_token_budget, sample_positions


def test_parse_all_token_budget() -> None:
    assert parse_token_budget("all") is None
    assert parse_token_budget("1000") == 1000


def test_sample_all_positions_without_random_index() -> None:
    documents = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    selected = sample_positions(documents, token_budget=None, seed=0)
    torch.testing.assert_close(selected[0], torch.tensor([0, 1]))
    torch.testing.assert_close(selected[1], torch.tensor([0, 1, 2]))
