from __future__ import annotations

import hashlib

import pytest

from icalens.cli.profile import _replay_dataset_source, parse_args


def test_parses_statistics_refresh_operation() -> None:
    args = parse_args(
        [
            "refresh-statistics",
            "--lens",
            "artifact",
            "--layers",
            "all",
            "--activations",
            "capture",
            "--max-tokens",
            "1000000",
        ]
    )

    assert args.operation == "refresh-statistics"
    assert args.activations.name == "capture"
    assert args.max_tokens == 1_000_000


def test_replay_dataset_source_accepts_matching_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"alpha"}\n')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()

    assert _replay_dataset_source({"path": str(dataset), "sha256": digest, "split": "train"}) == (
        str(dataset.resolve()),
        None,
    )


def test_replay_dataset_source_rejects_changed_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"changed"}\n')

    with pytest.raises(ValueError, match="checksum differs"):
        _replay_dataset_source({"path": str(dataset), "sha256": "0" * 64, "split": "train"})
