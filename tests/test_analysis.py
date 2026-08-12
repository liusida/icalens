from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icalens import ICALens
from icalens.analysis import (
    _chat_group_titles,
    _resolve_device,
    _resolve_model_and_tokenizer,
    _token_presentations,
)


class DummyTokenizer:
    model_max_length = 16

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        length = len(text.split())
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{value}" for value in ids]

    def decode(self, ids: list[int]) -> str:
        return f"t{ids[0]}"


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        values = input_ids.float()
        hidden = torch.stack((values, values.square(), values + 1), dim=-1)
        return SimpleNamespace(hidden_states=(hidden,))


def test_analyze_raw_text_returns_scores_and_energy(mixed_signals: np.ndarray) -> None:
    lens = ICALens(
        model_id="example/model",
        model_revision="abc",
        activation_site="hidden_states",
        layer_indexing="hidden_states",
    ).fit(mixed_signals, layer=0)
    result = lens.analyze(
        "one two three",
        layer=0,
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
    )
    assert result.tokens == ("t1", "t2", "t3")
    assert result.token_texts == ("t1", "t2", "t3")
    assert result.scores.shape == (3, 3)
    assert result.model == "example/model@abc"
    assert result.layer == 0
    assert result.input_text == "one two three"
    torch.testing.assert_close(result.energy.sum(dim=-1), torch.ones(3))


def test_auto_device_uses_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device(None) == "cpu"


def test_auto_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"
    assert _resolve_device("cpu") == "cpu"


def test_analysis_model_and_tokenizer_are_reused() -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    model = DummyModel()
    tokenizer = DummyTokenizer()
    lens._analysis_model = model
    lens._analysis_tokenizer = tokenizer
    lens._analysis_device = "cpu"

    resolved_model, resolved_tokenizer = _resolve_model_and_tokenizer(lens, None, None, "cpu")

    assert resolved_model is model
    assert resolved_tokenizer is tokenizer


def test_unload_model_clears_analysis_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    lens = ICALens(model_id="example/model", model_revision="abc")
    lens._analysis_model = DummyModel()
    lens._analysis_tokenizer = DummyTokenizer()
    lens._analysis_device = "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    lens.unload_model()

    assert lens._analysis_model is None
    assert lens._analysis_tokenizer is None
    assert lens._analysis_device is None


def test_invalid_token_fragment_uses_placeholder_and_contextual_tooltip() -> None:
    class FragmentTokenizer:
        def decode(self, ids: list[int]) -> str:
            if len(ids) > 1:
                return "你"
            return "�"

    texts, labels, tooltips = _token_presentations(
        FragmentTokenizer(), [99, 100], [0, 1], ("fragment-a", "fragment-b")
    )

    assert texts == ("�", "�")
    assert labels == ("<?>", "<?>")
    assert "Token ID 99 (0x0063)" in tooltips[0]
    assert "tokens 0–1 decode together as '你'" in tooltips[0]


def test_chat_tokens_are_grouped_with_their_message_template() -> None:
    rendered = "<start>user\nHi<end><start>assistant\nHello<end>"
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    pieces = [
        "<start>",
        "user",
        "\n",
        "Hi",
        "<end>",
        "<start>",
        "assistant",
        "\n",
        "Hello",
        "<end>",
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)

    groups = _chat_group_titles(rendered, messages, offsets, pieces)

    assert groups[:5] == ["User 1"] * 5
    assert groups[5:] == ["Assistant 1"] * 5
