#!/usr/bin/env python3
"""Build trajectory-wide fixed autointerpretability panels from existing caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from icalens.experiments._run import atomic_write_json

from prepare import EVALUATED_CHECKPOINTS, N_FEATURES, N_FRAGMENTS
from trajectory import parse_layers

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    ROOT / "runs/prepared",
    ROOT / "runs/qwen-layers-07-23/prepared",
)
DEFAULT_OUTPUT = ROOT / "runs/fixed-panel/prepared"
DEFAULT_LAYERS = (7, 15, 23, 31)
TOP_K = 20
EXAMPLES_PER_SPLIT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        type=Path,
        action="append",
        help="Prepared activation root; repeat when layers live in multiple roots",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink() and os.readlink(destination) == target:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(target)


def weighted_unique_document_sample(
    candidates: np.ndarray,
    weights: np.ndarray,
    document_ids: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    available = np.ones(len(candidates), dtype=bool)
    selected: list[int] = []
    used_documents: set[int] = set()
    while len(selected) < count:
        eligible = available & np.asarray(
            [int(document_ids[index]) not in used_documents for index in candidates]
        )
        positions = np.flatnonzero(eligible)
        if not len(positions):
            raise ValueError("not enough unique-document top fragments for a fixed panel")
        probabilities = weights[positions].astype(np.float64)
        probabilities /= probabilities.sum()
        chosen_position = int(rng.choice(positions, p=probabilities))
        chosen = int(candidates[chosen_position])
        selected.append(chosen)
        used_documents.add(int(document_ids[chosen]))
        available[chosen_position] = False
    return selected


def layer_panels(
    input_root: Path,
    layer: int,
    top_k: int,
    document_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids: list[int] | None = None
    membership = np.zeros((N_FRAGMENTS, N_FEATURES), dtype=np.uint8)
    positive_at_every_checkpoint = np.ones((N_FRAGMENTS, N_FEATURES), dtype=bool)
    candidate_positions: list[int] | None = None
    for iteration in EVALUATED_CHECKPOINTS:
        directory = input_root / f"iter-{iteration:03d}" / f"layer_{layer:02d}" / "ica"
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        accepted = selection["accepted"]
        current_ids = [int(row["feature"]) for row in accepted]
        current_positions = [int(row["candidate_position"]) for row in accepted]
        if ids is None:
            ids, candidate_positions = current_ids, current_positions
        elif current_ids != ids or current_positions != candidate_positions:
            raise ValueError(f"cohort identity changed at iteration {iteration}, layer {layer}")
        values = np.load(directory / "candidate_activations.npy", mmap_mode="r")
        maxima = np.asarray(values).max(axis=1)
        positive_at_every_checkpoint &= maxima > 0
        for position in range(N_FEATURES):
            top = np.argsort(-maxima[:, position], kind="stable")[:top_k]
            membership[top, position] += 1
    assert ids is not None and candidate_positions is not None

    accepted_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for position, feature in enumerate(ids):
        candidates = np.flatnonzero(membership[:, position])
        weights = membership[candidates, position]
        rng = np.random.default_rng(np.random.SeedSequence([layer, feature]))
        fixed_top = weighted_unique_document_sample(
            candidates, weights, document_ids, count=2 * EXAMPLES_PER_SPLIT, rng=rng
        )
        used_documents = {int(document_ids[index]) for index in fixed_top}
        permutation = rng.permutation(N_FRAGMENTS)
        fixed_random = [
            int(index)
            for index in permutation
            if membership[index, position] == 0
            and positive_at_every_checkpoint[index, position]
            and int(document_ids[index]) not in used_documents
        ][:EXAMPLES_PER_SPLIT]
        if len(fixed_random) != EXAMPLES_PER_SPLIT:
            raise ValueError(f"not enough fixed random fragments for layer {layer}, C{feature}")
        row = {
            "feature": feature,
            "candidate_position": candidate_positions[position],
            "train_top": fixed_top[:EXAMPLES_PER_SPLIT],
            "valid_top": fixed_top[EXAMPLES_PER_SPLIT:],
            "valid_random": fixed_random,
        }
        accepted_rows.append(row)
        audit_rows.append({
            "feature": feature,
            "union_size": int(len(candidates)),
            "train_top": [
                {"fragment": index, "top_20_checkpoint_count": int(membership[index, position])}
                for index in row["train_top"]
            ],
            "valid_top": [
                {"fragment": index, "top_20_checkpoint_count": int(membership[index, position])}
                for index in row["valid_top"]
            ],
            "valid_random": fixed_random,
        })
    return accepted_rows, {"layer": layer, "components": audit_rows}


def resolve_layer_sources(
    inputs: tuple[Path, ...], layers: tuple[int, ...]
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for layer in layers:
        matches = [
            root
            for root in inputs
            if (root / f"iter-000/layer_{layer:02d}/ica/selection.json").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one prepared input for layer {layer}; found {matches}"
            )
        result[layer] = matches[0]
    return result


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.top_k < 2 * EXAMPLES_PER_SPLIT:
        raise ValueError(f"--top-k must be at least {2 * EXAMPLES_PER_SPLIT}")
    inputs = tuple(
        path.expanduser().resolve()
        for path in (args.inputs if args.inputs is not None else DEFAULT_INPUTS)
    )
    layer_sources = resolve_layer_sources(inputs, layers)
    output = args.output.expanduser().resolve()
    fragments_paths = {
        (root / "iter-000/fragments.jsonl").resolve() for root in layer_sources.values()
    }
    if len({sha256(path) for path in fragments_paths}) != 1:
        raise ValueError(f"prepared inputs use different fragment corpora: {fragments_paths}")
    fragments_path = sorted(fragments_paths)[0]
    fragments = [json.loads(line) for line in fragments_path.read_text(encoding="utf-8").splitlines()]
    if len(fragments) != N_FRAGMENTS:
        raise ValueError(f"expected {N_FRAGMENTS} fragments: {fragments_path}")
    document_ids = np.asarray([int(row["document_index"]) for row in fragments])
    resolved = {
        "format": "icalens.autointerpretability-trajectory-fixed-panel",
        "schema_version": 1,
        "source_preparations": {
            str(layer): {
                "path": str(root),
                "preparation_run_sha256": sha256(root / "preparation-run.json"),
            }
            for layer, root in layer_sources.items()
        },
        "fragments": str(fragments_path),
        "fragments_sha256": sha256(fragments_path),
        "layers": list(layers),
        "checkpoints": list(EVALUATED_CHECKPOINTS),
        "top_k_per_checkpoint": args.top_k,
        "top_sampling": "weighted without replacement by checkpoint top-k membership",
        "split_sizes": {"train_top": 5, "valid_top": 5, "valid_random": 5},
        "seed_rule": "numpy SeedSequence([layer, persistent_component_id])",
        "document_disjoint": True,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    manifest = output / "fixed-panel-run.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("resolved") != resolved:
            raise ValueError(f"incompatible existing fixed-panel preparation: {manifest}")
        if previous.get("status") == "complete":
            print(f"PASS fixed panel already complete: {output}")
            return

    output.mkdir(parents=True, exist_ok=True)
    panels: dict[int, list[dict[str, Any]]] = {}
    audits = []
    for layer in layers:
        print(f"START fixed panel layer {layer}")
        panels[layer], audit = layer_panels(
            layer_sources[layer], layer, args.top_k, document_ids
        )
        audits.append(audit)
        print(f"PASS fixed panel layer {layer}")
    for iteration in EVALUATED_CHECKPOINTS:
        destination_iteration = output / f"iter-{iteration:03d}"
        relative_symlink(fragments_path, destination_iteration / "fragments.jsonl")
        for layer in layers:
            source_iteration = layer_sources[layer] / f"iter-{iteration:03d}"
            source_layer = source_iteration / f"layer_{layer:02d}" / "ica"
            destination_layer = destination_iteration / f"layer_{layer:02d}" / "ica"
            relative_symlink(
                (source_layer / "candidate_activations.npy").resolve(),
                destination_layer / "candidate_activations.npy",
            )
            atomic_write_json(
                destination_layer / "selection.json",
                {"candidate_ids": [row["feature"] for row in panels[layer]],
                 "accepted": panels[layer], "rejected": []},
            )
            atomic_write_json(
                destination_iteration / f"layer_{layer:02d}" / "prepared.json",
                {"format": "icalens.autointerpretability-prepared-layer",
                 "schema_version": 1, "layer": layer, "methods": ["ica"],
                 "n_fragments": N_FRAGMENTS, "n_features": N_FEATURES,
                 "selection_protocol": "trajectory-wide-fixed-panel"},
            )
        atomic_write_json(
            destination_iteration / "run.json",
            {"status": "prepared", "resolved": {**resolved, "iteration": iteration}},
        )
    atomic_write_json(output / "fixed-panel-audit.json", {"resolved": resolved, "layers": audits})
    atomic_write_json(manifest, {"status": "complete", "resolved": resolved})
    print(f"Wrote fixed evaluator input to {output}")


if __name__ == "__main__":
    main()
