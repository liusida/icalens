from __future__ import annotations

import hashlib

import pytest

from icalens.cli.profile import _replay_dataset_source


def test_replay_dataset_source_accepts_matching_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"alpha"}\n')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()

    assert _replay_dataset_source(
        {"path": str(dataset), "sha256": digest, "split": "train"}
    ) == (str(dataset.resolve()), None)


def test_replay_dataset_source_rejects_changed_local_file(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"changed"}\n')

    with pytest.raises(ValueError, match="checksum differs"):
        _replay_dataset_source(
            {"path": str(dataset), "sha256": "0" * 64, "split": "train"}
        )
