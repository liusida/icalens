"""SAE ERF validation and threshold-independent batching without model downloads."""

from types import SimpleNamespace

import pytest
import torch

from icalens.experiments import erf_suffix_sweep as sweep
from icalens.experiments._run import atomic_write_json
from icalens.experiments.erf_sae import load_checkpoint, validate_result


def test_zero_activation_and_competition_rank():
    rank, sign, recovered = sweep._recovery_from_scores(
        torch.zeros(1, 8), component=0, direction="positive", top_k=5
    )
    assert rank.item() == 1 and not sign.item() and not recovered.item()
    rank, _, recovered = sweep._recovery_from_scores(
        torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 5.0]]), component=5, direction="positive", top_k=5
    )
    assert rank.item() == 5 and recovered.item()


def test_checkpoint_identity(tmp_path):
    p = tmp_path / "checkpoint.json"
    assert load_checkpoint(p, {"k": 5}) is None
    atomic_write_json(p, {"identity": {"k": 5}, "value": 1})
    assert load_checkpoint(p, {"k": 5})["value"] == 1
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(p, {"k": 1})


def test_inactive_and_missing_threshold_validation():
    validate_result({"n_occurrences": 0, "status": "no_positive_examples"}, 0)
    with pytest.raises(ValueError, match="incorrect result thresholds"):
        validate_result({"n_occurrences": 1, "occurrences": [{}], "rank_thresholds": [5]}, 1)


def test_fixed_batches_across_thresholds_and_resume(monkeypatch):
    monkeypatch.setattr(sweep, "transformer_blocks", lambda model: [None])
    monkeypatch.setattr(sweep, "_suffix_schedule", lambda **kwargs: [1, 2, 3])
    monkeypatch.setattr(
        sweep,
        "_prepare_occurrence",
        lambda occurrence, occurrence_rank, **kwargs: {
            "occurrence": occurrence,
            "occurrence_rank": occurrence_rank,
            "content_ids": [1, 2, 3],
        },
    )
    monkeypatch.setattr(
        sweep,
        "_finish_component_result",
        lambda **kwargs: kwargs["occurrence_states"][1]["recoveries"],
    )
    prepared = {
        i: {"tail_direction": "positive", "occurrences": [{"absolute_score_rank": 1}]}
        for i in range(4)
    }
    lens = SimpleNamespace(
        metadata={"layers": {"0": {"fitting": {"provenance": {"document_framing": {}}}}}}
    )

    def run(thresholds, completed=()):
        seen, results = [], {}

        def measure(**kwargs):
            ids = tuple(c for c, _, _ in kwargs["batch"])
            length = kwargs["suffix_length"]
            seen.append((length, ids))
            return [
                (
                    c,
                    {
                        "occurrence_rank": 1,
                        "suffix_length": length,
                        "sign_matches_selected_tail": True,
                        "absolute_score_rank": 1 if length == 3 else (3 if c % 2 == 0 else 8),
                    },
                )
                for c in ids
            ]

        monkeypatch.setattr(sweep, "_measure_mixed_batch", measure)
        sweep._measure_layer(
            lens=lens,
            model=None,
            tokenizer=None,
            datasets={},
            token_cache={},
            layer=0,
            prepared_components=prepared,
            exact_suffix_length=10,
            rank_thresholds=thresholds,
            max_batch_size=2,
            batch_token_budget=100,
            device="cpu",
            stable_batches=True,
            completed_components=set(completed),
            checkpoint=lambda c, r: results.update({c: r}),
        )
        return seen, results

    a, single = run((5,))
    b, multi = run((1, 3, 5, 10, 15))
    assert all(ids in [(0, 1), (2, 3)] for _, ids in a + b)
    assert {c: r[5] for c, r in single.items()} == {c: r[5] for c, r in multi.items()}
    resumed, results = run((1, 3, 5, 10, 15), (0, 2, 999))
    assert all(ids in [(0, 1), (2, 3)] for _, ids in resumed)
    assert set(results) == {1, 3}
    assert results == {c: multi[c] for c in (1, 3)}
