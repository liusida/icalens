from __future__ import annotations

import json
from pathlib import Path

import torch

from icalens.experiments.erf_suffix_sweep import (
    METHOD,
    _adaptive_batch_size,
    _component_checkpoint_valid,
    _recovery_from_scores,
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


def test_component_checkpoint_validation(tmp_path: Path) -> None:
    path = tmp_path / "component.json"
    path.write_text(
        json.dumps(
            {
                "model_label": "gpt2",
                "layer": 2,
                "component": 5,
                "method": METHOD,
                "suffix_erf_median": 3.0,
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
        "model,layer,component,suffix_erf_mean\ngpt2,0,1,1.5\ngpt2,0,2,11.0\n"
    )

    outputs = render(
        experiment,
        output_prefix=experiment / "figures" / "suffix",
        panel_titles=None,
        force=False,
    )

    assert outputs == [
        experiment / "figures" / "suffix.png",
        experiment / "figures" / "suffix.pdf",
        experiment / "figures" / "suffix.txt",
    ]
    assert all(path.is_file() for path in outputs)
