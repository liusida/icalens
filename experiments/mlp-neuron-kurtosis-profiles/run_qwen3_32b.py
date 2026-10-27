#!/usr/bin/env python3
"""Run the MLP-neuron kurtosis profile experiment on Qwen3-32B Base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

import run as base
from icalens._activation_dataset import sample_metadata
from icalens.cli._status import log
from icalens.cli.fit_text import load_pile_documents, sample_positions
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
DEFAULT_OUTPUT = HERE / "runs/qwen3-32b-layer32"
DEFAULT_LAYER = 32
LAYER_COUNT = 64
HIDDEN_SIZE = 5120
MLP_WIDTH = 25600
STATISTICS_TOKENS = 1_000_000
PROFILE_TOKENS = 100_000
CONTEXT_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--profile-tokens", type=int, default=PROFILE_TOKENS)
    parser.add_argument("--top-k", type=int, default=base.DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-documents", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_statistics(path: Path) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        expected = {
            "count": (),
            "neuron": (MLP_WIDTH,),
            "mean": (MLP_WIDTH,),
            "variance": (MLP_WIDTH,),
            "skewness": (MLP_WIDTH,),
            "excess_kurtosis": (MLP_WIDTH,),
        }
        if any(key not in data or data[key].shape != shape for key, shape in expected.items()):
            return False
        return (
            int(data["count"]) == STATISTICS_TOKENS
            and np.array_equal(data["neuron"], np.arange(MLP_WIDTH))
            and all(
                np.isfinite(data[key]).all()
                for key in ("mean", "variance", "skewness", "excess_kurtosis")
            )
            and bool(np.all(data["variance"] >= 0.0))
        )


def main() -> None:
    args = parse_args()
    if not 0 <= args.layer < LAYER_COUNT:
        raise ValueError(f"--layer must be in 0..{LAYER_COUNT - 1}")
    if args.document_batch_size <= 0 or args.checkpoint_documents <= 0:
        raise ValueError("batch and checkpoint sizes must be positive")
    if not 0 < args.profile_tokens <= STATISTICS_TOKENS:
        raise ValueError(f"--profile-tokens must be in 1..{STATISTICS_TOKENS}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-32B")
    output = args.output.expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    probe = "Tokenizer framing probe."
    if tokenizer(probe, add_special_tokens=True)["input_ids"] != tokenizer(
        probe, add_special_tokens=False
    )["input_ids"]:
        raise ValueError("Qwen3 tokenizer unexpectedly adds special tokens to plain text")
    framing = {
        "requested": "model-tokenizer-natural",
        "strategy": "none",
        "token": None,
        "token_id": None,
        "included_in_fitting_samples": False,
        "policy_source": "experiment:tokenizer-add-special-tokens",
        "evidence_note": (
            "For Qwen3-32B plain text, tokenizer(..., add_special_tokens=True) adds "
            "neither BOS nor EOS. Documents are passed as raw token sequences."
        ),
    }
    documents = load_pile_documents(
        tokenizer,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        split=DATASET_SPLIT,
        text_field=TEXT_FIELD,
        candidate_token_budget=None,
        context_length=CONTEXT_LENGTH,
        document_framing=framing,
    )
    candidate_tokens = sum(
        int(document.candidate_positions.numel()) for document in documents
    )
    if candidate_tokens < STATISTICS_TOKENS:
        raise ValueError(
            f"dataset yielded only {candidate_tokens:,} candidate tokens; "
            f"need at least {STATISTICS_TOKENS:,}"
        )
    fitting_positions = sample_positions(
        documents, token_budget=STATISTICS_TOKENS, seed=args.seed
    )
    metadata = sample_metadata(documents, fitting_positions)
    if any(int(values.shape[0]) != STATISTICS_TOKENS for values in metadata.values()):
        raise ValueError("sample metadata does not contain exactly one million positions")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    profile_rows = torch.randperm(STATISTICS_TOKENS, generator=generator)[
        : args.profile_tokens
    ].sort().values
    all_rows = torch.arange(STATISTICS_TOKENS, dtype=torch.long)
    statistics_rows = base.rows_by_document(metadata, all_rows)
    occurrence_rows = base.rows_by_document(metadata, profile_rows)

    resolved = {
        "format": base.FORMAT,
        "schema_version": base.SCHEMA_VERSION,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "type": "base"},
        "model_title": "Qwen3 32B Base",
        "population_source": "reconstructed-directly-from-pinned-text-dataset",
        "layer": args.layer,
        "layer_count": LAYER_COUNT,
        "layer_indexing": "transformer_blocks_zero_based",
        "hidden_size": HIDDEN_SIZE,
        "mlp_width": MLP_WIDTH,
        "quantity": "post-gating MLP coefficient entering down_proj",
        "direction": "corresponding column of W_downproj",
        "statistics_tokens": STATISTICS_TOKENS,
        "statistics_sampling": "seeded uniform without replacement from candidate rows",
        "statistics_seed": args.seed,
        "profile_tokens": args.profile_tokens,
        "profile_sampling": "seeded uniform without replacement from fitting rows",
        "profile_seed": args.seed,
        "dominant_sign": "sign of population skewness",
        "selection": "lowest 3, seeded random 3 above median excluding extremes, highest 3",
        "top_k_occurrences": args.top_k,
        "source_dataset": {
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "text_field": TEXT_FIELD,
        "candidate_tokens": candidate_tokens,
        "context_length": CONTEXT_LENGTH,
        "document_framing": framing,
    }
    source_state = source_provenance()
    run = ResumableRun.open(
        output=output,
        resolved=resolved,
        source=source_state,
        status="running",
    )
    warn_if_dirty(source_state)
    statistics_path = output / "neuron-statistics.npz"
    profiles_path = output / "selected-neuron-profiles.json"
    statistics_complete = validate_statistics(statistics_path)
    profiles_complete = base.validate_profiles(profiles_path, top_k=args.top_k)
    if profiles_complete and not statistics_complete:
        raise ValueError("profiles exist without valid neuron statistics")
    completed = {
        phase
        for phase, complete in (
            ("statistics", statistics_complete),
            ("occurrences", profiles_complete),
        )
        if complete
    }
    if len(completed) == 2:
        log("All requested phases are already complete and compatible.")
        run.set_status("complete", complete=True)
        return

    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · Qwen3-32B MLP neuron kurtosis profiles",
        completed=len(completed),
        total=2,
        completed_unit_ids=completed,
        unit_label="phases",
        source_dirty=bool(source_state.get("dirty")),
    ) as display:
        warn_if_dirty(source_state)
        display.phase("Loading Qwen3-32B", layer=args.layer)
        model = load_model_to_cuda(
            AutoModelForCausalLM,
            MODEL_ID,
            revision=MODEL_REVISION,
            device=args.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            touch="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        if int(model.config.hidden_size) != HIDDEN_SIZE:
            raise ValueError(f"unexpected hidden size: {model.config.hidden_size}")
        projection = base.down_projection(model, args.layer)
        if tuple(projection.weight.shape) != (HIDDEN_SIZE, MLP_WIDTH):
            raise ValueError(f"unexpected down_proj shape: {tuple(projection.weight.shape)}")
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Qwen3 tokenizer has no pad token")

        if "statistics" not in completed:
            display.phase("Measuring population moments", layer=args.layer)
            base.run_statistics(
                model=model,
                projection=projection,
                documents=documents,
                selected_rows=statistics_rows,
                pad_token_id=int(pad_token_id),
                document_batch_size=args.document_batch_size,
                checkpoint_documents=args.checkpoint_documents,
                checkpoint_path=output / "statistics.partial.npz",
                output_path=statistics_path,
            )
            completed.add("statistics")
            display.complete_unit("statistics", refresh=True)
        else:
            log(f"Reusing completed neuron statistics: {statistics_path}")

        with np.load(statistics_path, allow_pickle=False) as stored:
            statistics = {key: stored[key].copy() for key in stored.files}
        selected_neurons, groups = base.select_neurons(
            statistics["excess_kurtosis"], seed=args.seed
        )
        if "occurrences" not in completed:
            display.phase("Finding dominant-tail occurrences", layer=args.layer)
            base.run_occurrences(
                model=model,
                projection=projection,
                documents=documents,
                metadata=metadata,
                selected_rows=occurrence_rows,
                selected_neurons=selected_neurons,
                groups=groups,
                statistics=statistics,
                tokenizer=tokenizer,
                pad_token_id=int(pad_token_id),
                document_batch_size=args.document_batch_size,
                checkpoint_documents=args.checkpoint_documents,
                top_k=args.top_k,
                checkpoint_path=output / "occurrences.partial.npz",
                partial_output_path=output / "selected-neuron-profiles.partial.json",
                output_path=profiles_path,
            )
            completed.add("occurrences")
            display.complete_unit("occurrences", refresh=True)
        else:
            log(f"Reusing completed selected-neuron profiles: {profiles_path}")
        display.phase("Finalizing run")
    base.atomic_write_json(
        output / "summary.json",
        {
            "status": "complete",
            "statistics": str(statistics_path),
            "profiles": str(profiles_path),
            "selected_neurons": selected_neurons.tolist(),
            "selection_groups": groups,
        },
    )
    run.set_status("complete", complete=True)
    log(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
