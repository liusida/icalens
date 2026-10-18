#!/usr/bin/env python3
"""Add longer tokenwise-scored contexts to completed neuron profiles."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

import run as experiment
from icalens._activation_dataset import ActivationDataset
from icalens.cli._status import log
from icalens.experiments._run import atomic_write_json

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/main"
DEFAULT_INPUT = DEFAULT_RUN / "selected-neuron-profiles.json"
DEFAULT_OUTPUT = DEFAULT_RUN / "selected-neuron-profiles-context.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--left-context", type=int, default=48)
    parser.add_argument("--right-context", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Replace existing output.")
    return parser.parse_args()


def load_inputs(run_root: Path, profile_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    resolved = run.get("resolved", {})
    if (
        resolved.get("format") != experiment.FORMAT
        or resolved.get("schema_version") != experiment.SCHEMA_VERSION
        or run.get("status") != "complete"
    ):
        raise ValueError(f"not a completed compatible run: {run_root}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 9:
        raise ValueError("expected nine completed selected-neuron profiles")
    if any(len(profile.get("occurrences", [])) != 20 for profile in profiles):
        raise ValueError("each selected neuron must have 20 occurrences")
    return run, payload


def requested_positions(
    profiles: list[dict[str, Any]],
    documents: list[Any],
    *,
    left_context: int,
    right_context: int,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[tuple[int, int], int]]:
    positions_by_document: dict[int, set[int]] = defaultdict(set)
    for profile in profiles:
        for occurrence in profile["occurrences"]:
            document = int(occurrence["source_index"])
            target = int(occurrence["position"])
            if not 0 <= document < len(documents):
                raise ValueError(f"occurrence references unavailable document {document}")
            ids = documents[document].input_ids
            if not 0 <= target < len(ids) or int(ids[target]) != int(occurrence["token_id"]):
                raise ValueError(f"occurrence token mismatch at document {document}, {target}")
            start = max(0, target - left_context)
            end = min(len(ids), target + right_context + 1)
            positions_by_document[document].update(range(start, end))
    pair_to_row: dict[tuple[int, int], int] = {}
    selected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    next_row = 0
    for document in sorted(positions_by_document):
        positions = sorted(positions_by_document[document])
        rows = list(range(next_row, next_row + len(positions)))
        next_row += len(positions)
        pair_to_row.update({(document, position): row for position, row in zip(positions, rows)})
        selected[document] = (
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(rows, dtype=torch.long),
        )
    return selected, pair_to_row


def enrich(
    profiles: list[dict[str, Any]],
    documents: list[Any],
    tokenizer: Any,
    scores: np.ndarray,
    pair_to_row: dict[tuple[int, int], int],
    neuron_to_column: dict[int, int],
    *,
    left_context: int,
    right_context: int,
) -> list[dict[str, Any]]:
    enriched = copy.deepcopy(profiles)
    for profile in enriched:
        neuron = int(profile["neuron"])
        column = neuron_to_column[neuron]
        for occurrence in profile["occurrences"]:
            document = int(occurrence["source_index"])
            target = int(occurrence["position"])
            ids = documents[document].input_ids
            start = max(0, target - left_context)
            end = min(len(ids), target + right_context + 1)
            occurrence["context_start_position"] = start
            occurrence["context_end_position"] = end
            occurrence["context_tokens"] = [
                {
                    "position": position,
                    "token_id": int(ids[position]),
                    "text": tokenizer.decode(
                        [int(ids[position])], clean_up_tokenization_spaces=False
                    ),
                    "score": float(scores[pair_to_row[document, position], column]),
                    "target": position == target,
                }
                for position in range(start, end)
            ]
    return enriched


def main() -> None:
    args = parse_args()
    if min(args.left_context, args.right_context) < 0:
        raise ValueError("context lengths must be nonnegative")
    if args.document_batch_size <= 0:
        raise ValueError("--document-batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default enrichment pass")
    run_root = args.run.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace it: {output_path}")
    run, payload = load_inputs(run_root, input_path)
    resolved = run["resolved"]
    dataset = ActivationDataset(Path(resolved["activation_dataset"]))
    tokenizer = AutoTokenizer.from_pretrained(
        dataset.model["repo_id"],
        revision=dataset.model.get("revision"),
        trust_remote_code=True,
    )
    documents = experiment.load_documents(dataset, tokenizer)
    profiles = payload["profiles"]
    selected, pair_to_row = requested_positions(
        profiles,
        documents,
        left_context=args.left_context,
        right_context=args.right_context,
    )
    neurons = np.asarray([int(profile["neuron"]) for profile in profiles], dtype=np.int64)
    neuron_to_column = {int(neuron): column for column, neuron in enumerate(neurons)}
    log(
        f"Scoring {len(pair_to_row):,} context tokens across "
        f"{len(selected):,} occurrence documents."
    )
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        dataset.model["repo_id"],
        revision=dataset.model.get("revision"),
        device=args.device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    projection = experiment.down_projection(model, int(resolved["layer"]))
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    score_rows = np.empty((len(pair_to_row), len(neurons)), dtype=np.float32)
    document_ids = sorted(selected)
    for start in range(0, len(document_ids), args.document_batch_size):
        batch_ids = document_ids[start : start + args.document_batch_size]
        values, rows = experiment.capture_coefficients(
            model,
            projection,
            documents,
            batch_ids,
            selected,
            pad_token_id=int(pad_token_id),
        )
        score_rows[rows.numpy()] = values[:, neurons].float().cpu().numpy()
        log(f"Context documents {start + len(batch_ids):,}/{len(document_ids):,}")
    enriched_profiles = enrich(
        profiles,
        documents,
        tokenizer,
        score_rows,
        pair_to_row,
        neuron_to_column,
        left_context=args.left_context,
        right_context=args.right_context,
    )
    atomic_write_json(
        output_path,
        {
            "status": "complete",
            "source": str(input_path),
            "layer": int(resolved["layer"]),
            "left_context_tokens": args.left_context,
            "right_context_tokens": args.right_context,
            "scored_context_tokens": len(pair_to_row),
            "occurrence_documents": len(selected),
            "profiles": enriched_profiles,
        },
    )
    log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
