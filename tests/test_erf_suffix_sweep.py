from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from icalens.experiments.erf_suffix_sweep import (
    METHOD,
    _adaptive_batch_size,
    _component_checkpoint_valid,
    _finish_component_result,
    _recovery_estimate,
    _recovery_from_scores,
    _suffix_schedule,
)
from icalens.experiments.erf_suffix_sweep_figure import render


def test_recovery_requires_rank_and_selected_tail_sign() -> None:
    scores = torch.tensor(
        [
            [5.0, 4.0, 3.0],
            [5.0, -6.0, 3.0],
            [5.0, 0.5, 3.0],
        ]
    )

    ranks, sign_ok, recovered = _recovery_from_scores(
        scores, component=1, direction="positive", top_k=2
    )

    assert ranks.tolist() == [2, 1, 3]
    assert sign_ok.tolist() == [True, False, True]
    assert recovered.tolist() == [True, False, False]


def test_adaptive_batch_size_uses_a_fixed_token_budget() -> None:
    assert _adaptive_batch_size(1, max_batch_size=64, token_budget=64) == 64
    assert _adaptive_batch_size(2, max_batch_size=64, token_budget=64) == 32
    assert _adaptive_batch_size(3, max_batch_size=64, token_budget=64) == 21
    assert _adaptive_batch_size(10, max_batch_size=64, token_budget=64) == 6


def test_suffix_schedule_is_exact_through_ten_then_doubles() -> None:
    assert _suffix_schedule(exact_suffix_length=10, maximum_context=95) == [
        *range(1, 11),
        20,
        40,
        80,
    ]
    assert _suffix_schedule(exact_suffix_length=10, maximum_context=16) == [*range(1, 11)]


def test_recovery_estimate_uses_geometric_midpoint_after_exact_sweep() -> None:
    assert _recovery_estimate(
        lower=9, upper=10, exact_suffix_length=10, source="measured_suffix"
    )["erf_estimate"] == 10
    estimated = _recovery_estimate(
        lower=10, upper=20, exact_suffix_length=10, source="measured_suffix"
    )
    assert estimated["exact"] is False
    assert estimated["erf_estimate"] == pytest.approx(200**0.5)


def test_component_result_records_threshold_specific_eligibility() -> None:
    occurrence = {
        "source_index": 4,
        "position": 7,
        "token_id": 9,
        "text": " token",
        "context": "some token",
        "score": 8.0,
        "energy": 0.2,
    }
    state = {
        "item": {"occurrence_rank": 1, "occurrence": occurrence, "content_ids": [1] * 20},
        "full_context_rank": 4,
        "eligible": {5, 10, 15},
        "recoveries": {
            threshold: _recovery_estimate(
                lower=10, upper=20, exact_suffix_length=10, source="measured_suffix"
            )
            for threshold in (5, 10, 15)
        },
        "observations": [],
    }

    result = _finish_component_result(
        layer=2,
        component=5,
        direction="positive",
        occurrence_states={1: state},
        exact_suffix_length=10,
        rank_thresholds=(1, 3, 5, 10, 15),
    )

    assert result["threshold_results"]["3"]["n_eligible"] == 0
    assert result["threshold_results"]["5"]["n_eligible"] == 1
    assert result["occurrences"][0]["thresholds"]["3"] == {"eligible": False}


def test_component_checkpoint_validation(tmp_path: Path) -> None:
    path = tmp_path / "component.json"
    path.write_text(
        json.dumps(
            {
                "model_label": "gpt2",
                "layer": 2,
                "component": 5,
                "method": METHOD,
                    "threshold_results": {"15": {"suffix_erf_mean": 3.0}},
                "n_occurrences": 1,
                "occurrences": [{}],
            }
        )
    )

    assert _component_checkpoint_valid(path, label="gpt2", layer=2, component=5)
    assert not _component_checkpoint_valid(path, label="gpt2", layer=2, component=6)


def test_suffix_sweep_figure_writes_all_outputs(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "resolved": {
                    "components_per_layer": 2,
                    "rank_thresholds": [1, 3, 5, 10, 15],
                    "seed": 0,
                    "lens_order": ["gpt2"],
                    "lenses": {
                        "gpt2": {
                            "model": {"repo_id": "openai-community/gpt2"},
                            "layers": [0],
                        }
                    },
                },
            }
        )
    )
    (experiment / "summary.csv").write_text(
        "model,layer,component,top_k,suffix_erf_mean\n"
        "gpt2,0,1,15,1.5\n"
        "gpt2,0,2,15,20.0\n"
    )

    outputs = render(
        experiment,
        output_prefix=experiment / "figures" / "suffix",
        panel_titles=None,
        top_k=15,
        force=False,
    )

    assert outputs == [
        experiment / "figures" / "suffix.png",
        experiment / "figures" / "suffix.pdf",
        experiment / "figures" / "suffix.txt",
    ]
    assert all(path.is_file() for path in outputs)
