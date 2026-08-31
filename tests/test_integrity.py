from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from icalens._activation_dataset import ActivationDatasetWriter, sample_metadata
from icalens.cli.integrity import (
    _audit_official_experiments,
    _capture_command,
    _compare_activations,
    _discover_official_experiments,
    _sample_indices,
)


@dataclass
class _Document:
    input_ids: torch.Tensor


def _activation_fixture(path, *, changed: bool = False) -> None:
    documents = [_Document(torch.tensor([10, 11, 12, 13]))]
    selected = {0: torch.arange(4)}
    writer = ActivationDatasetWriter(
        path,
        model={"repo_id": "owner/model", "revision": "abc", "type": "base"},
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        layers=(6,),
        sample_count=4,
        hidden_size=2,
        dtype=torch.bfloat16,
        provenance={"dataset": {"repo_id": "owner/data", "revision": "def"}},
        samples=sample_metadata(documents, selected),
    )
    values = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    if changed:
        values[2, 1] += 1
    with writer.group((6,)) as sink:
        sink.append({6: values})
    writer.finish()


def test_sample_indices_are_deterministic_and_sorted() -> None:
    first = _sample_indices(100, 12, 0)
    second = _sample_indices(100, 12, 0)
    assert torch.equal(first, second)
    assert torch.all(first[1:] > first[:-1])


def test_activation_comparison_detects_sampled_value_change(tmp_path) -> None:
    reference = tmp_path / "reference"
    identical = tmp_path / "identical"
    changed = tmp_path / "changed"
    _activation_fixture(reference)
    _activation_fixture(identical)
    _activation_fixture(changed, changed=True)

    assert _compare_activations(reference, identical, layer=6, rows=4, seed=0)["status"] == "pass"
    assert _compare_activations(reference, changed, layer=6, rows=4, seed=0)["status"] == "fail"


def test_capture_command_pins_recorded_revisions(tmp_path) -> None:
    resolved = {
        "capture": {
            "kind": "text",
            "model": "owner/model",
            "model_revision": "model-sha",
            "dataset": "owner/data",
            "dataset_revision": "dataset-sha",
            "split": "train",
            "context_length": 128,
            "candidate_tokens": 2000,
            "token_budget": 1000,
            "seed": 0,
            "text_field": "text",
            "document_framing": "prepend-eos",
        }
    }

    command = _capture_command(resolved, tmp_path / "capture", 6)

    assert command[command.index("--model-revision") + 1] == "model-sha"
    assert command[command.index("--dataset-revision") + 1] == "dataset-sha"
    assert command[command.index("--layers") + 1] == "6"


def test_official_experiment_audit_selects_model_and_layer(tmp_path) -> None:
    result = tmp_path / "reconstruction" / "official" / "results" / "gpt2"
    result.mkdir(parents=True)
    resolved = {
        "experiment": "reconstruction",
        "model_id": "owner/model",
        "layers": [6],
    }
    (result / "run.json").write_text(
        json.dumps({"status": "complete", "resolved": resolved})
    )
    (result / "results.json").write_text(
        json.dumps(
            {
                "experiment": resolved,
                "rows": [{"layer": 6, "k": "1", "method": "ica", "nmse_mean": 0.2}],
            }
        )
    )

    artifacts = _discover_official_experiments(tmp_path, model_id="owner/model", layer=6)
    audit = _audit_official_experiments(artifacts)

    assert len(artifacts) == 1
    assert audit["status"] == "pass"
    assert audit["artifacts"][0]["matching_rows"] == 1


def test_official_experiment_audit_rejects_non_finite_metric(tmp_path) -> None:
    result = tmp_path / "sparse-probing" / "official" / "results" / "gpt2"
    result.mkdir(parents=True)
    resolved = {
        "experiment": "saebench-sparse-probing",
        "model_id": "owner/model",
        "layers": [6],
    }
    (result / "run.json").write_text(
        json.dumps({"status": "complete", "resolved": resolved})
    )
    (result / "results.json").write_text(
        json.dumps(
            {
                "experiment": resolved,
                "rows": [
                    {
                        "layer": 6,
                        "k": 1,
                        "method": "ica",
                        "mean_probe_accuracy": float("nan"),
                    }
                ],
            }
        )
    )

    artifacts = _discover_official_experiments(tmp_path, model_id="owner/model", layer=6)

    assert _audit_official_experiments(artifacts)["status"] == "fail"
