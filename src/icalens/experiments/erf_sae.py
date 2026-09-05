"""Official resumable SAE suffix ERF using the shared ICA sweep engine."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from icalens._activation_dataset import ActivationDataset
from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.experiments import erf_gradient, erf_suffix_sweep
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._saebench_worker import SAEFeatureEncoder
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import _prepare_layer_baselines, _resolve_baselines

ROOT = Path.cwd()
DEFAULT_CACHE = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/gemma-2-2b-pile10k-1m"
)
CHUNK = 32768


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SAEReadout:
    """Adapt only score computation/provenance, without copying the ERF algorithm."""

    def __init__(self, cache, encoder, layer):
        self.model_id = cache.model["repo_id"]
        self.model_revision = cache.model["revision"]
        self.encoder = encoder
        self.layer = layer
        self.metadata = {"layers": {str(layer): {"fitting": {"provenance": cache.provenance}}}}

    def transform(self, hidden, *, layer):
        if layer != self.layer:
            raise ValueError("incorrect SAE layer")
        return self.encoder.encode(hidden)


def load_checkpoint(path, identity):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("identity") != identity:
        raise ValueError(f"incompatible checkpoint: {path}")
    return value


def profile(cache, encoder, ids, output, identity, display):
    """Checkpoint cumulative top occurrences at fixed activation-shard boundaries."""
    path = output / "profile-progress.json"
    state = load_checkpoint(path, identity)
    count = identity["occurrences"]
    end = 0 if state is None else state["end"]
    if not 0 <= end <= cache.sample_count or (end != cache.sample_count and end % CHUNK):
        raise ValueError("invalid profile checkpoint offset")
    values = torch.zeros((len(ids), count), device="cuda")
    indices = torch.full((len(ids), count), -1, dtype=torch.long, device="cuda")
    if state is not None:
        values = torch.tensor(state["values"], device="cuda")
        indices = torch.tensor(state["indices"], device="cuda")
        if values.shape != (len(ids), count) or indices.shape != values.shape:
            raise ValueError("invalid profile checkpoint shape")
    hidden = cache.layer(identity["layer"])
    ids_tensor = torch.tensor(ids, device="cuda")
    for start in range(end, cache.sample_count, CHUNK):
        stop = min(start + CHUNK, cache.sample_count)
        display.phase("Profiling SAE features", layer=identity["layer"], dataset="Pile-10k")
        for batch in range(start, stop, 512):
            batch_end = min(batch + 512, stop)
            codes = encoder.encode(hidden[batch:batch_end].to("cuda", dtype=torch.float32))
            selected = codes[:, ids_tensor].T
            if not torch.isfinite(selected).all() or (selected < 0).any():
                raise ValueError("SAE scores must be finite and nonnegative")
            new_values, local_indices = selected.topk(min(count, batch_end - batch), dim=1)
            combined = torch.cat((values, new_values), dim=1)
            combined_indices = torch.cat((indices, local_indices + batch), dim=1)
            values, order = combined.topk(count, dim=1)
            indices = combined_indices.gather(1, order)
        atomic_write_json(
            path,
            {
                "identity": identity,
                "end": stop,
                "values": values.tolist(),
                "indices": indices.tolist(),
            },
        )
        display.complete_unit(f"profile-{start // CHUNK}")
        log(f"Profiled {stop}/{cache.sample_count} candidate activations")
    return values.cpu(), indices.cpu()


def prepare_feature(feature, values, indices, cache, encoder, tokenizer, layer):
    samples = cache.samples()
    hidden = cache.layer(layer)
    occurrences = []
    for score, index in zip(values.tolist(), indices.tolist(), strict=True):
        if score <= 0 or index < 0:
            continue
        codes = encoder.encode(hidden[index : index + 1].to("cuda", dtype=torch.float32))[0]
        rank = int((codes > codes[feature]).sum()) + 1
        token = int(samples["token_id"][index])
        position = int(samples["position"][index])
        if not 1 <= position < 1024:
            raise ValueError("expected content positions after BOS within context 1024")
        occurrences.append(
            {
                "source_index": int(samples["document_index"][index]),
                "position": position,
                "token_id": token,
                "text": tokenizer.decode([token]),
                "context": "",
                "score": float(codes[feature]),
                "energy": float(codes[feature].square() / codes.square().sum()),
                "absolute_score_rank": rank,
                "activation_row": index,
            }
        )
    return {"component": feature, "tail_direction": "positive", "occurrences": occurrences}


def validate_result(value, expected_count, thresholds=(1, 3, 5, 10, 15)):
    if value.get("n_occurrences") != expected_count:
        raise ValueError("incomplete feature checkpoint")
    if expected_count == 0:
        if value.get("status") != "no_positive_examples":
            raise ValueError("missing inactive-feature status")
        return
    if len(value.get("occurrences", [])) != expected_count:
        raise ValueError("missing occurrence results")
    if value.get("rank_thresholds") != list(thresholds):
        raise ValueError("incorrect result thresholds")
    for threshold in thresholds:
        estimates = []
        recovered = 0
        for occurrence in value["occurrences"]:
            recovery = occurrence["thresholds"][str(threshold)]
            erf = recovery["erf_estimate"]
            if not math.isfinite(erf) or not 1 <= erf <= occurrence["content_tokens"]:
                raise ValueError("invalid ERF estimate")
            estimates.append(erf)
            recovered += recovery["recovered"]
        summary = value["threshold_results"][str(threshold)]
        if not np.isclose(np.mean(estimates), summary["suffix_erf_mean"]):
            raise ValueError("invalid component aggregation")
        if recovered != summary["n_recovered"]:
            raise ValueError("invalid recovery count")


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--occurrences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rank-thresholds", type=erf_suffix_sweep._parse_rank_thresholds, default=(1, 3, 5, 10, 15)
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audit-features", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.audit_features < 1:
        parser.error("batch-size and audit-features must be positive")
    cache = ActivationDataset(args.cache)
    if (
        cache.manifest["activation_site"] != "resid_post"
        or args.layer not in cache.available_layers
    ):
        raise ValueError("requires a cached resid_post layer")
    if (
        cache.provenance["context_length"] != 1024
        or cache.provenance["document_framing"]["strategy"] not in {"prepend-bos", "prepend-eos"}
    ):
        raise ValueError("requires the recorded 1024-token document-framing protocol")
    baseline = _prepare_layer_baselines(
        _resolve_baselines(cache.model["repo_id"], "sae"), layer=args.layer
    )
    width = int(baseline["sae"]["width"])
    if not 1 <= args.features <= width or not 1 <= args.occurrences <= 512:
        raise ValueError("invalid feature/example count")
    ids = sorted(
        np.random.default_rng(args.seed).choice(width, args.features, replace=False).tolist()
    )
    dependencies = [
        Path(__file__),
        Path(inspect.getfile(erf_suffix_sweep)),
        Path(inspect.getfile(erf_gradient)),
        Path(inspect.getfile(SAEFeatureEncoder)),
        cache.path / "activations.json",
        cache.path / cache.manifest["samples_file"],
        cache.path / cache.manifest["layers"][str(args.layer)]["file"],
        Path(baseline["sae"]["weights_file"]),
    ]
    identity = {
        "protocol": "sae-suffix-multirank-stable-batches-v1",
        "model": cache.model,
        "layer": args.layer,
        "features": ids,
        "occurrences": args.occurrences,
        "seed": args.seed,
        "candidate_tokens": cache.sample_count,
        "baseline": baseline,
        "provenance": cache.provenance,
        "rank_thresholds": list(args.rank_thresholds),
        "exact_suffix_length": 10,
        "max_batch_size": args.batch_size,
        "batch_token_budget": 4096,
        "stable_batches": True,
        "audit_features": args.audit_features,
        "score_convention": "checkpoint encoder activation multiplied by decoder norm",
        "selection": "uniform feature IDs without replacement; no resampling inactive features",
        "dependencies": {str(p.resolve()): digest(p) for p in dependencies},
    }
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        return
    output = args.output.resolve()
    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=identity, source=source, status="running")
    progress = load_checkpoint(output / "profile-progress.json", identity)
    completed = set()
    if progress:
        completed.update(f"profile-{i}" for i in range(math.ceil(progress["end"] / CHUNK)))
    prepared = {}
    for feature in ids:
        p = load_checkpoint(output / f"prepared/F{feature:05d}.json", identity)
        if p:
            prepared[feature] = p["feature"]
            completed.add(f"prepare-{feature}")
        result = load_checkpoint(output / f"results/F{feature:05d}.json", identity)
        if result:
            if not p:
                raise ValueError("result has no prepared feature")
            validate_result(result, len(p["feature"]["occurrences"]), args.rank_thresholds)
            completed.add(f"measure-{feature}")
    total = math.ceil(cache.sample_count / CHUNK) + 2 * len(ids)
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title=f"SAE suffix ERF · {cache.model['repo_id']} · layer {args.layer}",
            total=total,
            completed=len(completed),
            completed_unit_ids=completed,
            source_dirty=source.get("dirty"),
            unit_label="checkpoints",
        ) as display:
            warn_if_dirty(source)
            if len(completed) == total:
                log("All validated checkpoints complete; skipping model and encoder loading")
            else:
                display.phase("Loading SAE", layer=args.layer)
                encoder = SAEFeatureEncoder(
                    {
                        "hidden_size": cache.hidden_size,
                        "layer": args.layer,
                        "saebench_model_name": cache.model["repo_id"],
                        "baselines": baseline,
                    },
                    device="cuda",
                    dtype=torch.float32,
                ).eval()
                values, indices = profile(cache, encoder, ids, output, identity, display)
                tokenizer = AutoTokenizer.from_pretrained(
                    cache.model["repo_id"], revision=cache.model["revision"]
                )
                for i, feature in enumerate(ids):
                    if feature not in prepared:
                        display.phase("Preparing positive examples", component=f"F{feature}")
                        prepared[feature] = prepare_feature(
                            feature, values[i], indices[i], cache, encoder, tokenizer, args.layer
                        )
                        atomic_write_json(
                            output / f"prepared/F{feature:05d}.json",
                            {"identity": identity, "feature": prepared[feature]},
                        )
                        display.complete_unit(f"prepare-{feature}")
                pending = {f: prepared[f] for f in ids if f"measure-{f}" not in completed}

                def checkpoint(feature, result):
                    result["identity"] = identity
                    validate_result(
                        result, len(prepared[feature]["occurrences"]), args.rank_thresholds
                    )
                    atomic_write_json(output / f"results/F{feature:05d}.json", result)
                    display.complete_unit(f"measure-{feature}")
                    log(
                        f"Completed F{feature}: "
                        f"{result.get('threshold_results', result.get('status'))}"
                    )

                for feature in list(pending):
                    if not pending[feature]["occurrences"]:
                        checkpoint(
                            feature,
                            {
                                "component": feature,
                                "n_occurrences": 0,
                                "status": "no_positive_examples",
                            },
                        )
                        del pending[feature]
                if pending:
                    adapter = SAEReadout(cache, encoder, args.layer)
                    display.phase("Loading language model", layer=args.layer)
                    model = erf_gradient._load_model(adapter, device="cuda")
                    datasets, token_cache = {}, {}
                    # Multi-feature smoke audit, not a claim of bitwise cache equality.
                    audits = []
                    for feature in list(pending)[: args.audit_features]:
                        example = pending[feature]["occurrences"][0]
                        item = erf_gradient._prepare_occurrence(
                            example,
                            occurrence_rank=1,
                            provenance=cache.provenance,
                            tokenizer=tokenizer,
                            datasets=datasets,
                            token_cache=token_cache,
                        )
                        sequence = [int(cache.provenance["document_framing"]["token_id"])] + item[
                            "content_ids"
                        ]
                        live = erf_suffix_sweep._suffix_scores(
                            lens=adapter,
                            model=model,
                            block=transformer_blocks(model)[args.layer],
                            tokenizer=tokenizer,
                            layer=args.layer,
                            sequences=[sequence],
                            device="cuda",
                        )[0]
                        live_rank = int((live > live[feature]).sum()) + 1
                        audit = {
                            "feature": feature,
                            "stored_score": example["score"],
                            "live_score": float(live[feature]),
                            "stored_rank": example["absolute_score_rank"],
                            "live_rank": live_rank,
                            "threshold_disagreements": [
                                t
                                for t in args.rank_thresholds
                                if (live_rank <= t and float(live[feature]) > 0)
                                != (example["absolute_score_rank"] <= t)
                            ],
                        }
                        audits.append(audit)
                    atomic_write_json(output / "full-prefix-audit.json", audits)
                    failures = [
                        a
                        for a in audits
                        if not math.isclose(
                            a["stored_score"], a["live_score"], rel_tol=0.05, abs_tol=0.01
                        )
                        or a["threshold_disagreements"]
                    ]
                    if failures:
                        raise ValueError(
                            "Full-prefix cache audit failed; "
                            f"inspect full-prefix-audit.json: {failures}"
                        )
                    log(f"Full-prefix audit passed for {len(audits)} features")
                    display.phase(
                        f"Measuring suffix recovery {args.rank_thresholds}", layer=args.layer
                    )
                    erf_suffix_sweep._measure_layer(
                        lens=adapter,
                        model=model,
                        tokenizer=tokenizer,
                        datasets=datasets,
                        token_cache=token_cache,
                        layer=args.layer,
                        prepared_components={f: p for f, p in prepared.items() if p["occurrences"]},
                        exact_suffix_length=10,
                        rank_thresholds=args.rank_thresholds,
                        max_batch_size=args.batch_size,
                        batch_token_budget=4096,
                        device="cuda",
                        checkpoint=checkpoint,
                        stable_batches=True,
                        completed_components={f for f in prepared if f not in pending},
                    )
            results = [load_checkpoint(output / f"results/F{f:05d}.json", identity) for f in ids]
            active = [r for r in results if r["n_occurrences"]]
            atomic_write_json(
                output / "summary.json",
                {
                    "identity": identity,
                    "sampled_features": len(ids),
                    "features_with_examples": len(active),
                    "features_without_examples": len(ids) - len(active),
                    "features": [
                        {"feature": r["component"], "threshold_results": r["threshold_results"]}
                        for r in active
                    ],
                },
            )
            run.set_status("complete", complete=True)
            log(f"Complete: {len(active)}/{len(ids)} sampled features had positive examples")
    except BaseException:
        run.set_status("interrupted")
        raise


if __name__ == "__main__":
    main()
