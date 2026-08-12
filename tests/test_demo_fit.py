from __future__ import annotations

import torch

from icalens.cli.fit_text import parse_args, parse_token_budget, sample_positions


def test_parse_all_token_budget() -> None:
    assert parse_token_budget("all") is None
    assert parse_token_budget("1000") == 1000


def test_parse_text_source_options() -> None:
    args = parse_args(
        [
            "--model",
            "owner/model",
            "--dataset",
            "owner/dataset",
            "--split",
            "validation",
            "--text-field",
            "body",
            "--context-length",
            "512",
        ]
    )
    assert args.model == "owner/model"
    assert args.dataset == "owner/dataset"
    assert args.split == "validation"
    assert args.text_field == "body"
    assert args.context_length == 512


def test_sample_all_positions_without_random_index() -> None:
    documents = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    selected = sample_positions(documents, token_budget=None, seed=0)
    torch.testing.assert_close(selected[0], torch.tensor([0, 1]))
    torch.testing.assert_close(selected[1], torch.tensor([0, 1, 2]))
