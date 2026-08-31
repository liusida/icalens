from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from icalens.experiments.erf_gradient import (
    METHOD,
    _component_checkpoint_valid,
    _gradient_statistics,
    _stable_seed,
)
from icalens.experiments.erf_gradient_figure import render


def test_gradient_statistics_use_geometric_distance() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.7])

    result, distances = _gradient_statistics(probabilities)

    assert distances.tolist() == [3.0, 2.0, 1.0]
    expected = math.exp(0.1 * math.log(3) + 0.2 * math.log(2))
    assert result["gradient_erf"] == pytest.approx(expected)
    assert result["arithmetic_gradient_erf"] == pytest.approx(1.4)
    assert result["influence_radius_50"] == 1
    assert result["influence_radius_90"] == 2


def test_stable_seed_depends_on_label_and_layer() -> None:
    assert _stable_seed(7, "gpt2", 3) == _stable_seed(7, "gpt2", 3)
    assert _stable_seed(7, "gpt2", 3) != _stable_seed(7, "gpt2", 4)
    assert _stable_seed(7, "gpt2", 3) != _stable_seed(7, "gemma2", 3)


def test_component_checkpoint_validation(tmp_path: Path) -> None:
    path = tmp_path / "component.json"
    path.write_text(
        json.dumps(
            {
                "model_label": "gpt2",
                "layer": 2,
                "component": 5,
                "method": METHOD,
                "gradient_erf_median": 1.5,
                "n_occurrences": 1,
                "occurrences": [{}],
            }
        )
    )

    assert _component_checkpoint_valid(path, label="gpt2", layer=2, component=5, method=METHOD)
    assert not _component_checkpoint_valid(path, label="gpt2", layer=2, component=6, method=METHOD)


def test_erf_gradient_figure_writes_all_outputs(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    run = {
        "status": "complete",
        "resolved": {
            "components_per_layer": 2,
            "seed": 3,
            "lens_order": ["gpt2"],
            "lenses": {"gpt2": {"model": {"repo_id": "openai-community/gpt2", "revision": "r"}}},
        },
    }
    (experiment / "run.json").write_text(json.dumps(run))
    with (experiment / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "layer", "component", "gradient_erf_median"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"model": "gpt2", "layer": 0, "component": 1, "gradient_erf_median": 1.5},
                {"model": "gpt2", "layer": 0, "component": 2, "gradient_erf_median": 3.5},
            ]
        )

    outputs = render(
        experiment,
        output_prefix=tmp_path / "figure" / "erf-gradient",
        panel_titles=None,
        force=False,
    )

    assert outputs == [
        tmp_path / "figure" / "erf-gradient.png",
        tmp_path / "figure" / "erf-gradient.pdf",
        tmp_path / "figure" / "erf-gradient.txt",
    ]
    assert all(path.is_file() for path in outputs)
