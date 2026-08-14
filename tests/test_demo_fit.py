from __future__ import annotations

import torch

from icalens.cli.fit_text import (
    TextDocument,
    parse_args,
    parse_token_budget,
    resolve_document_framing,
    sample_positions,
)


def test_parse_all_token_budget() -> None:
    assert parse_token_budget("all") is None
    assert parse_token_budget("1000") == 1000


def test_parse_all_candidate_tokens() -> None:
    args = parse_args(["--candidate-tokens", "all"])
    assert args.candidate_tokens is None


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


def test_sample_positions_excludes_document_prefix() -> None:
    documents = [
        TextDocument(
            input_ids=torch.tensor([99, 1, 2]),
            candidate_positions=torch.tensor([1, 2]),
        ),
        TextDocument(
            input_ids=torch.tensor([99, 3]),
            candidate_positions=torch.tensor([1]),
        ),
    ]
    selected = sample_positions(documents, token_budget=None, seed=0)
    torch.testing.assert_close(selected[0], torch.tensor([1, 2]))
    torch.testing.assert_close(selected[1], torch.tensor([1]))


def test_auto_document_framing_uses_exact_gemma_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_token": "<bos>",
            "bos_token_id": 2,
            "eos_token": "<eos>",
            "eos_token_id": 1,
        },
    )()
    framing = resolve_document_framing(
        tokenizer, "auto", model_id="google/gemma-2-2b"
    )
    assert framing["strategy"] == "prepend-bos"
    assert framing["token_id"] == 2
    assert framing["included_in_fitting_samples"] is False


def test_auto_document_framing_uses_exact_gpt2_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_token": "<|endoftext|>",
            "bos_token_id": 50_256,
            "eos_token": "<|endoftext|>",
            "eos_token_id": 50_256,
        },
    )()
    framing = resolve_document_framing(
        tokenizer, "auto", model_id="openai-community/gpt2"
    )
    assert framing["strategy"] == "prepend-eos"
    assert framing["token_id"] == 50_256
    assert framing["registry_model_entry"] == "openai-community/gpt2"


def test_explicit_document_framing_does_not_require_registry_entry() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {"eos_token": "<eos>", "eos_token_id": 7},
    )()
    framing = resolve_document_framing(
        tokenizer, "prepend-eos", model_id="owner/unknown-model"
    )
    assert framing["strategy"] == "prepend-eos"
    assert framing["policy_source"] == "explicit-cli"
