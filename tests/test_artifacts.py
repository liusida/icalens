from __future__ import annotations

import json

import numpy as np
import pytest

from icalens import ArtifactError, ICALens


def make_lens() -> ICALens:
    return ICALens(
        base_model="example/model",
        base_model_revision="abc123",
        activation_site="resid_post",
    )


def test_save_and_local_load_round_trip(tmp_path, mixed_signals: np.ndarray) -> None:
    original = make_lens().fit(mixed_signals, layer=6, n_components=2, max_iter=500)
    expected = original.transform(mixed_signals[:10], layer=6)

    artifact_dir = original.save(tmp_path / "artifact")
    loaded = ICALens.from_pretrained(artifact_dir)

    assert (artifact_dir / "icalens.json").is_file()
    assert (artifact_dir / "README.md").is_file()
    assert (artifact_dir / "artifacts/resid_post/layer_06.safetensors").is_file()
    assert loaded.available_layers == (6,)
    assert loaded.metadata["base_model"]["revision"] == "abc123"
    np.testing.assert_allclose(loaded.transform(mixed_signals[:10], layer=6), expected)


def test_save_replaces_existing_artifact(tmp_path, mixed_signals: np.ndarray) -> None:
    target = tmp_path / "artifact"
    make_lens().fit(mixed_signals, layer=1, n_components=2).save(target)
    make_lens().fit(mixed_signals, layer=3, n_components=2).save(target)
    loaded = ICALens.from_pretrained(target)
    assert loaded.available_layers == (3,)


def test_rejects_unsupported_format_version(tmp_path) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "icalens.json").write_text(
        json.dumps({"format": "icalens", "format_version": 999}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="format version"):
        ICALens.from_pretrained(target)


def test_manifest_does_not_expose_mutable_internal_state(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1, n_components=2)
    metadata = lens.metadata
    metadata["layers"].clear()
    assert lens.available_layers == (1,)
