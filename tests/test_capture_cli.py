from __future__ import annotations

from icalens.cli.capture import _parse_args, _text_config


def test_text_capture_parses_skip_initial_tokens() -> None:
    args = _parse_args(
        "text",
        ["--output", "activations", "--skip-initial-tokens", "2"],
    )

    assert args.skip_initial_tokens == 2


def test_text_config_accepts_flat_causal_config() -> None:
    config = type("Config", (), {"num_hidden_layers": 12, "hidden_size": 768})()

    assert _text_config(config) is config


def test_text_config_unwraps_composite_config() -> None:
    text_config = type(
        "TextConfig", (), {"num_hidden_layers": 32, "hidden_size": 4096}
    )()
    config = type("Config", (), {"text_config": text_config})()

    assert _text_config(config) is text_config
