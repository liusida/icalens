from __future__ import annotations

import numpy as np
import pytest
import torch

from icalens import ICALens, NotFittedError


def make_lens() -> ICALens:
    return ICALens(
        model_id="example/model",
        model_revision="abc123",
        activation_site="resid_post",
    )


def test_instruct_model_metadata() -> None:
    lens = ICALens(
        model_id="example/model-instruct",
        model_revision="def456",
        model_type="instruct",
    )
    assert lens.model_id == "example/model-instruct"
    assert lens.model_revision == "def456"
    assert lens.model_type == "instruct"
    assert lens.metadata["model"]["type"] == "instruct"


def test_legacy_constructor_names_remain_compatible() -> None:
    lens = ICALens(base_model="example/model", base_model_revision="abc123")
    assert lens.model_id == "example/model"
    assert lens.base_model == lens.model_id
    assert lens.base_model_revision == lens.model_revision


def test_rejects_invalid_or_conflicting_model_arguments() -> None:
    with pytest.raises(ValueError, match="model_type"):
        ICALens(model_id="example/model", model_revision="abc123", model_type="chat")
    with pytest.raises(ValueError, match="not both"):
        ICALens(
            model_id="example/model",
            base_model="legacy/model",
            model_revision="abc123",
        )


def test_fit_transform_and_inverse(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=6, random_state=3, max_iter=500)

    scores = lens.transform(mixed_signals, layer=6)
    reconstructed = lens.inverse_transform(scores, layer=6)
    normalized = mixed_signals / np.maximum(
        np.linalg.norm(mixed_signals, axis=-1, keepdims=True), lens.norm_eps
    )

    assert lens.available_layers == (6,)
    assert lens.hidden_size == 3
    assert scores.shape == mixed_signals.shape
    np.testing.assert_allclose(reconstructed, normalized, atol=2e-5)


def test_leading_dimensions_are_preserved(mixed_signals: np.ndarray) -> None:
    values = mixed_signals[:30].reshape(3, 10, 3)
    lens = make_lens().fit(mixed_signals, layer=2, n_components=2)
    assert lens.transform(values, layer=2).shape == (3, 10, 2)


def test_energy_is_per_position_fraction(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=2)
    scores = lens.transform(mixed_signals[:10], layer=2)
    energy = lens.energy(scores)
    np.testing.assert_allclose(energy.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(energy, scores**2 / np.sum(scores**2, axis=-1, keepdims=True))


def test_fit_records_detached_json_provenance(mixed_signals: np.ndarray) -> None:
    provenance = {"dataset": {"repo_id": "example/data", "revision": "abc"}}
    lens = make_lens().fit(mixed_signals, layer=2, provenance=provenance)
    provenance["dataset"]["repo_id"] = "changed"
    fitting = lens.metadata["layers"]["2"]["fitting"]
    assert fitting["source_scaling"] == "none"
    assert fitting["provenance"]["dataset"]["repo_id"] == "example/data"


def test_torch_transform_preserves_tensor_properties(mixed_signals: np.ndarray) -> None:
    values = torch.from_numpy(mixed_signals)
    lens = make_lens().fit(values, layer=2, n_components=2)
    scores = lens.transform(values[:10], layer=2)
    reconstructed = lens.inverse_transform(scores, layer=2)
    assert isinstance(scores, torch.Tensor)
    assert scores.dtype == values.dtype
    assert scores.device == values.device
    assert reconstructed.shape == (10, 3)


def test_deflation_algorithm(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(
        mixed_signals, layer=2, n_components=2, algorithm="deflation", max_iter=500
    )
    assert lens.transform(mixed_signals[:10], layer=2).shape == (10, 2)


def test_blockwise_fit_matches_single_batch(mixed_signals: np.ndarray) -> None:
    single_batch = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=5,
        batch_size=len(mixed_signals),
    )
    blockwise = make_lens().fit(
        mixed_signals,
        layer=2,
        n_components=3,
        max_iter=5,
        batch_size=37,
    )
    np.testing.assert_allclose(
        blockwise.transform(mixed_signals, layer=2),
        single_batch.transform(mixed_signals, layer=2),
        atol=1e-5,
        rtol=1e-5,
    )
    fitting = blockwise.metadata["layers"]["2"]["fitting"]
    assert fitting["batch_size"] == 37
    assert fitting["memory_strategy"] == "blockwise_multi_pass"


def test_repeated_fit_builds_collection(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1, n_components=2)
    returned = lens.fit(mixed_signals, layer=4, n_components=3)
    assert returned is lens
    assert lens.available_layers == (1, 4)


def test_unknown_layer_and_shape_are_clear(mixed_signals: np.ndarray) -> None:
    lens = make_lens().fit(mixed_signals, layer=1)
    with pytest.raises(NotFittedError, match="available layers: 1"):
        lens.transform(mixed_signals, layer=2)
    with pytest.raises(ValueError, match="expected 3"):
        lens.transform(np.ones((4, 2), dtype=np.float32), layer=1)


def test_empty_lens_cannot_be_saved(tmp_path) -> None:
    with pytest.raises(NotFittedError):
        make_lens().save(tmp_path / "artifact")


def test_full_components_require_one_more_sample_than_features() -> None:
    values = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="centering limits the data rank"):
        make_lens().fit(values, layer=1, n_components=3)
