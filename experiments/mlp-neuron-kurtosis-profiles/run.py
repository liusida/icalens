#!/usr/bin/env python3
"""Measure and profile Qwen MLP neurons selected by excess kurtosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._activation_dataset import ActivationDataset
from icalens._capture import transformer_blocks
from icalens.cli._status import log
from icalens.cli.fit_text import TextDocument, load_pile_documents
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_ACTIVATIONS = Path(
    "/home/liusida/Expansion/research/ICA-data/icalens-activations/"
    "qwen3.5-9b-base-pile10k-1m"
)
DEFAULT_OUTPUT = HERE / "runs/main"
EXPECTED_MODEL = "Qwen/Qwen3.5-9B-Base"
EXPECTED_SAMPLE_COUNT = 1_000_000
EXPECTED_HIDDEN_SIZE = 4096
DEFAULT_LAYER = 16
DEFAULT_PROFILE_TOKENS = 100_000
DEFAULT_TOP_K = 20
FORMAT = "icalens.mlp_neuron_kurtosis_profiles"
SCHEMA_VERSION = 1


class _CaptureComplete(Exception):
    """Stop a forward immediately after the down-projection input is captured."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--profile-tokens", type=int, default=DEFAULT_PROFILE_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-documents", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            raise ValueError(f"temporary checkpoint failed validation: {temporary}")
    os.replace(temporary, path)


def validate_inputs(dataset: ActivationDataset, layer: int) -> None:
    expected = {
        "model": (dataset.model.get("repo_id"), EXPECTED_MODEL),
        "sample_count": (dataset.sample_count, EXPECTED_SAMPLE_COUNT),
        "hidden_size": (dataset.hidden_size, EXPECTED_HIDDEN_SIZE),
        "activation_site": (dataset.manifest.get("activation_site"), "resid_post"),
        "layer_indexing": (
            dataset.manifest.get("layer_indexing"),
            "transformer_blocks_zero_based",
        ),
    }
    differences = [
        f"{key}: {actual!r} != {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if layer not in dataset.available_layers:
        differences.append(f"layer {layer} is absent from the activation cache")
    if differences:
        raise ValueError("incompatible activation dataset: " + "; ".join(differences))


def load_documents(dataset: ActivationDataset, tokenizer: Any) -> list[TextDocument]:
    provenance = dataset.manifest["provenance"]
    source = provenance["dataset"]
    if "repo_id" not in source:
        raise ValueError("this experiment currently requires a pinned Hub text dataset")
    return load_pile_documents(
        tokenizer,
        dataset_id=str(source["repo_id"]),
        dataset_revision=str(source["revision"]),
        split=str(source["split"]),
        text_field=str(provenance.get("text_field", "text")),
        candidate_token_budget=int(provenance["candidate_tokens"]),
        context_length=int(provenance["context_length"]),
        document_framing=provenance.get("document_framing"),
    )


def validate_documents(
    dataset: ActivationDataset, documents: list[TextDocument]
) -> dict[str, torch.Tensor]:
    metadata = dataset.samples()
    document_index = metadata["document_index"].long()
    position = metadata["position"].long()
    token_id = metadata["token_id"].long()
    for row in range(dataset.sample_count):
        doc = int(document_index[row])
        pos = int(position[row])
        if doc >= len(documents) or pos >= len(documents[doc].input_ids):
            raise ValueError(f"activation row {row} points outside replayed documents")
        if int(documents[doc].input_ids[pos]) != int(token_id[row]):
            raise ValueError(f"activation row {row} does not match replayed tokenization")
    log(f"Validated {dataset.sample_count:,} cached rows against replayed tokenization.")
    return metadata


def rows_by_document(
    metadata: dict[str, torch.Tensor], rows: torch.Tensor
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    grouped_positions: dict[int, list[int]] = defaultdict(list)
    grouped_rows: dict[int, list[int]] = defaultdict(list)
    document_indices = metadata["document_index"]
    positions = metadata["position"]
    for row in rows.tolist():
        document = int(document_indices[row])
        grouped_positions[document].append(int(positions[row]))
        grouped_rows[document].append(int(row))
    return {
        document: (
            torch.tensor(grouped_positions[document], dtype=torch.long),
            torch.tensor(grouped_rows[document], dtype=torch.long),
        )
        for document in grouped_positions
    }


def down_projection(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    blocks = transformer_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"layer must be in 0..{len(blocks) - 1}")
    mlp = getattr(blocks[layer], "mlp", None)
    projection = getattr(mlp, "down_proj", None)
    if not isinstance(projection, torch.nn.Module) or not hasattr(projection, "weight"):
        raise ValueError(f"layer {layer} has no supported mlp.down_proj")
    return projection


def capture_coefficients(
    model: torch.nn.Module,
    projection: torch.nn.Module,
    documents: list[TextDocument],
    batch_documents: list[int],
    selected: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(documents[index].input_ids) for index in batch_documents]
    width = max(lengths)
    input_ids = torch.full(
        (len(batch_documents), width), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for batch_index, document_index in enumerate(batch_documents):
        ids = documents[document_index].input_ids
        input_ids[batch_index, : len(ids)] = ids
        attention_mask[batch_index, : len(ids)] = 1
    model_device = next(model.parameters()).device
    captured: list[torch.Tensor] = []
    row_ids: list[torch.Tensor] = []

    def hook(_: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        values = inputs[0]
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise TypeError("down_proj input must have shape [batch, tokens, neurons]")
        for batch_index, document_index in enumerate(batch_documents):
            positions, rows = selected[document_index]
            captured.append(values[batch_index].index_select(0, positions.to(values.device)))
            row_ids.append(rows)
        raise _CaptureComplete

    handle = projection.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            try:
                model(
                    input_ids=input_ids.to(model_device),
                    attention_mask=attention_mask.to(model_device),
                    use_cache=False,
                )
            except _CaptureComplete:
                pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("down_proj hook did not capture neuron coefficients")
    return torch.cat(captured, dim=0), torch.cat(row_ids, dim=0)


def empty_moments(width: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "mean": torch.zeros(width, dtype=torch.float64),
        "m2": torch.zeros(width, dtype=torch.float64),
        "m3": torch.zeros(width, dtype=torch.float64),
        "m4": torch.zeros(width, dtype=torch.float64),
    }


def update_moments(state: dict[str, torch.Tensor | int], values: torch.Tensor) -> None:
    """Merge one batch's central moments into a stable population accumulator."""
    batch_count = int(values.shape[0])
    if batch_count == 0:
        return
    batch_mean_gpu = values.float().mean(dim=0)
    centered = values.float() - batch_mean_gpu
    batch_mean = batch_mean_gpu.double().cpu()
    batch_m2 = centered.square().sum(dim=0, dtype=torch.float64).cpu()
    batch_m3 = centered.pow(3).sum(dim=0, dtype=torch.float64).cpu()
    batch_m4 = centered.pow(4).sum(dim=0, dtype=torch.float64).cpu()
    old_count = int(state["count"])
    if old_count == 0:
        state.update(
            count=batch_count,
            mean=batch_mean,
            m2=batch_m2,
            m3=batch_m3,
            m4=batch_m4,
        )
        return
    new_count = old_count + batch_count
    old_mean = state["mean"]
    old_m2 = state["m2"]
    old_m3 = state["m3"]
    old_m4 = state["m4"]
    assert all(isinstance(value, torch.Tensor) for value in (old_mean, old_m2, old_m3, old_m4))
    delta = batch_mean - old_mean
    a, b, n = float(old_count), float(batch_count), float(new_count)
    state["mean"] = old_mean + delta * (b / n)
    state["m2"] = old_m2 + batch_m2 + delta.square() * (a * b / n)
    state["m3"] = (
        old_m3
        + batch_m3
        + delta.pow(3) * (a * b * (a - b) / n**2)
        + 3.0 * delta * (a * batch_m2 - b * old_m2) / n
    )
    state["m4"] = (
        old_m4
        + batch_m4
        + delta.pow(4) * (a * b * (a * a - a * b + b * b) / n**3)
        + 6.0 * delta.square() * (a * a * batch_m2 + b * b * old_m2) / n**2
        + 4.0 * delta * (a * batch_m3 - b * old_m3) / n
    )
    state["count"] = new_count


def save_moment_checkpoint(
    path: Path, state: dict[str, torch.Tensor | int], next_document: int
) -> None:
    atomic_save_npz(
        path,
        count=np.asarray(int(state["count"]), dtype=np.int64),
        next_document=np.asarray(next_document, dtype=np.int64),
        mean=np.asarray(state["mean"]),
        m2=np.asarray(state["m2"]),
        m3=np.asarray(state["m3"]),
        m4=np.asarray(state["m4"]),
    )


def load_moment_checkpoint(
    path: Path, width: int
) -> tuple[dict[str, torch.Tensor | int], int]:
    if not path.is_file():
        return empty_moments(width), 0
    with np.load(path, allow_pickle=False) as data:
        state: dict[str, torch.Tensor | int] = {
            "count": int(data["count"]),
            "mean": torch.from_numpy(data["mean"].copy()),
            "m2": torch.from_numpy(data["m2"].copy()),
            "m3": torch.from_numpy(data["m3"].copy()),
            "m4": torch.from_numpy(data["m4"].copy()),
        }
        if any(np.asarray(state[key]).shape != (width,) for key in ("mean", "m2", "m3", "m4")):
            raise ValueError(f"invalid moment checkpoint: {path}")
        return state, int(data["next_document"])


def finalize_statistics(state: dict[str, torch.Tensor | int]) -> dict[str, np.ndarray]:
    count = int(state["count"])
    if count < 2:
        raise ValueError("at least two neuron observations are required")
    mean = torch.as_tensor(state["mean"])
    variance = torch.as_tensor(state["m2"]) / count
    safe = variance.clamp_min(torch.finfo(torch.float64).eps)
    skewness = (torch.as_tensor(state["m3"]) / count) / safe.pow(1.5)
    excess = (torch.as_tensor(state["m4"]) / count) / safe.square() - 3.0
    return {
        "count": np.asarray(count, dtype=np.int64),
        "neuron": np.arange(mean.numel(), dtype=np.int64),
        "mean": mean.numpy(),
        "variance": variance.numpy(),
        "skewness": skewness.numpy(),
        "excess_kurtosis": excess.numpy(),
    }


def select_neurons(
    excess_kurtosis: np.ndarray, *, seed: int
) -> tuple[np.ndarray, list[str]]:
    if excess_kurtosis.ndim != 1 or len(excess_kurtosis) < 9:
        raise ValueError("expected at least nine neuron kurtosis values")
    order = np.argsort(excess_kurtosis, kind="stable")
    lowest = order[:3]
    highest = order[-3:][::-1]
    median = float(np.median(excess_kurtosis))
    excluded = set(lowest.tolist()) | set(highest.tolist())
    candidates = np.asarray(
        [
            index
            for index in range(len(excess_kurtosis))
            if excess_kurtosis[index] > median and index not in excluded
        ],
        dtype=np.int64,
    )
    if len(candidates) < 3:
        raise ValueError("fewer than three non-extreme neurons lie above the median")
    sampled = np.random.default_rng(seed).choice(candidates, size=3, replace=False)
    selected = np.concatenate((lowest, sampled, highest)).astype(np.int64)
    groups = ["lowest"] * 3 + ["above_median_random"] * 3 + ["highest"] * 3
    return selected, groups


def validate_statistics(path: Path) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        expected = {
            "count": (),
            "neuron": (12288,),
            "mean": (12288,),
            "variance": (12288,),
            "skewness": (12288,),
            "excess_kurtosis": (12288,),
        }
        if any(key not in data or data[key].shape != shape for key, shape in expected.items()):
            return False
        if int(data["count"]) != EXPECTED_SAMPLE_COUNT:
            return False
        if not np.array_equal(data["neuron"], np.arange(12288)):
            return False
        return all(
            np.isfinite(data[key]).all()
            for key in ("mean", "variance", "skewness", "excess_kurtosis")
        ) and bool(np.all(data["variance"] >= 0.0))


def validate_profiles(path: Path, *, top_k: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 9:
        return False
    expected_groups = ["lowest"] * 3 + ["above_median_random"] * 3 + ["highest"] * 3
    return all(
        profile.get("selection_group") == group
        and profile.get("dominant_sign") in {"positive", "negative"}
        and isinstance(profile.get("occurrences"), list)
        and len(profile["occurrences"]) == top_k
        for profile, group in zip(profiles, expected_groups, strict=True)
    )


def run_statistics(
    *,
    model: torch.nn.Module,
    projection: torch.nn.Module,
    documents: list[TextDocument],
    selected_rows: dict[int, tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int,
    document_batch_size: int,
    checkpoint_documents: int,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    width = int(projection.weight.shape[1])
    state, next_document = load_moment_checkpoint(checkpoint_path, width)
    document_ids = sorted(selected_rows)
    if next_document > len(document_ids):
        raise ValueError(f"invalid statistics checkpoint position: {next_document}")
    for start in range(next_document, len(document_ids), document_batch_size):
        batch_ids = document_ids[start : start + document_batch_size]
        values, _ = capture_coefficients(
            model,
            projection,
            documents,
            batch_ids,
            selected_rows,
            pad_token_id=pad_token_id,
        )
        update_moments(state, values)
        end = start + len(batch_ids)
        if end == len(document_ids) or end % checkpoint_documents < document_batch_size:
            save_moment_checkpoint(checkpoint_path, state, end)
        log(
            f"Statistics documents {end:,}/{len(document_ids):,}; "
            f"tokens {int(state['count']):,}/1,000,000"
        )
    arrays = finalize_statistics(state)
    if int(arrays["count"]) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"statistics captured {int(arrays['count']):,} rows, expected 1,000,000")
    atomic_save_npz(output_path, **arrays)
    checkpoint_path.unlink(missing_ok=True)


def empty_top_state(neuron_count: int, top_k: int) -> dict[str, np.ndarray]:
    return {
        "priority": np.full((neuron_count, top_k), -np.inf, dtype=np.float64),
        "score": np.zeros((neuron_count, top_k), dtype=np.float64),
        "row": np.full((neuron_count, top_k), -1, dtype=np.int64),
    }


def update_top_state(
    state: dict[str, np.ndarray],
    values: torch.Tensor,
    rows: torch.Tensor,
    signs: np.ndarray,
) -> None:
    scores = values.detach().float().cpu().numpy().astype(np.float64, copy=False)
    row_values = rows.numpy()
    for column, sign in enumerate(signs):
        priority = scores[:, column] * float(sign)
        merged_priority = np.concatenate((state["priority"][column], priority))
        merged_scores = np.concatenate((state["score"][column], scores[:, column]))
        merged_rows = np.concatenate((state["row"][column], row_values))
        retained = np.argpartition(merged_priority, -state["priority"].shape[1])[
            -state["priority"].shape[1] :
        ]
        retained = retained[np.argsort(merged_priority[retained])[::-1]]
        state["priority"][column] = merged_priority[retained]
        state["score"][column] = merged_scores[retained]
        state["row"][column] = merged_rows[retained]


def save_occurrence_checkpoint(
    path: Path,
    state: dict[str, np.ndarray],
    *,
    next_document: int,
    selected_neurons: np.ndarray,
    signs: np.ndarray,
) -> None:
    atomic_save_npz(
        path,
        next_document=np.asarray(next_document, dtype=np.int64),
        selected_neurons=selected_neurons,
        signs=signs,
        priority=state["priority"],
        score=state["score"],
        row=state["row"],
    )


def load_occurrence_checkpoint(
    path: Path,
    *,
    selected_neurons: np.ndarray,
    signs: np.ndarray,
    top_k: int,
) -> tuple[dict[str, np.ndarray], int]:
    if not path.is_file():
        return empty_top_state(len(selected_neurons), top_k), 0
    with np.load(path, allow_pickle=False) as stored:
        expected_shape = (len(selected_neurons), top_k)
        if (
            not np.array_equal(stored["selected_neurons"], selected_neurons)
            or not np.array_equal(stored["signs"], signs)
            or any(stored[key].shape != expected_shape for key in ("priority", "score", "row"))
        ):
            raise ValueError(f"incompatible occurrence checkpoint: {path}")
        state = {
            key: stored[key].copy() for key in ("priority", "score", "row")
        }
        return state, int(stored["next_document"])


def record_for_row(
    row: int,
    metadata: dict[str, torch.Tensor],
    documents: list[TextDocument],
    tokenizer: Any,
) -> dict[str, Any]:
    document = int(metadata["document_index"][row])
    position = int(metadata["position"][row])
    token_id = int(metadata["token_id"][row])
    ids = documents[document].input_ids
    start, end = max(0, position - 4), min(len(ids), position + 5)
    context_ids = ids[start:end].tolist()
    target_offset = position - start
    prefix = tokenizer.decode(context_ids[:target_offset], clean_up_tokenization_spaces=False)
    through = tokenizer.decode(
        context_ids[: target_offset + 1], clean_up_tokenization_spaces=False
    )
    return {
        "activation_row": row,
        "source_index": document,
        "position": position,
        "token_id": token_id,
        "token": str(tokenizer.convert_ids_to_tokens(token_id)),
        "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
        "context": tokenizer.decode(context_ids, clean_up_tokenization_spaces=False),
        "context_target_start": len(prefix),
        "context_target_end": len(through),
    }


def occurrence_payload(
    *,
    state: dict[str, np.ndarray],
    selected_neurons: np.ndarray,
    groups: list[str],
    signs: np.ndarray,
    statistics: dict[str, np.ndarray],
    metadata: dict[str, torch.Tensor],
    documents: list[TextDocument],
    tokenizer: Any,
    processed_documents: int,
    total_documents: int,
    status: str,
) -> dict[str, Any]:
    profiles = []
    for column, (neuron, group, sign) in enumerate(
        zip(selected_neurons.tolist(), groups, signs.tolist(), strict=True)
    ):
        occurrences = []
        for score, row in zip(state["score"][column], state["row"][column], strict=True):
            if row < 0:
                continue
            occurrence = record_for_row(int(row), metadata, documents, tokenizer)
            occurrence["score"] = float(score)
            occurrences.append(occurrence)
        profiles.append(
            {
                "neuron": int(neuron),
                "down_projection_direction": f"column {neuron} of W_downproj",
                "selection_group": group,
                "mean": float(statistics["mean"][neuron]),
                "variance": float(statistics["variance"][neuron]),
                "skewness": float(statistics["skewness"][neuron]),
                "excess_kurtosis": float(statistics["excess_kurtosis"][neuron]),
                "dominant_sign": "positive" if sign > 0 else "negative",
                "occurrences": occurrences,
            }
        )
    return {
        "status": status,
        "processed_documents": processed_documents,
        "total_documents": total_documents,
        "profiles": profiles,
    }


def run_occurrences(
    *,
    model: torch.nn.Module,
    projection: torch.nn.Module,
    documents: list[TextDocument],
    metadata: dict[str, torch.Tensor],
    selected_rows: dict[int, tuple[torch.Tensor, torch.Tensor]],
    selected_neurons: np.ndarray,
    groups: list[str],
    statistics: dict[str, np.ndarray],
    tokenizer: Any,
    pad_token_id: int,
    document_batch_size: int,
    checkpoint_documents: int,
    top_k: int,
    checkpoint_path: Path,
    partial_output_path: Path,
    output_path: Path,
) -> None:
    document_ids = sorted(selected_rows)
    signs = np.where(statistics["skewness"][selected_neurons] >= 0.0, 1, -1)
    state, next_document = load_occurrence_checkpoint(
        checkpoint_path,
        selected_neurons=selected_neurons,
        signs=signs,
        top_k=top_k,
    )
    if next_document > len(document_ids):
        raise ValueError(f"invalid occurrence checkpoint position: {next_document}")
    for start in range(next_document, len(document_ids), document_batch_size):
        batch_ids = document_ids[start : start + document_batch_size]
        values, rows = capture_coefficients(
            model,
            projection,
            documents,
            batch_ids,
            selected_rows,
            pad_token_id=pad_token_id,
        )
        update_top_state(state, values[:, selected_neurons], rows, signs)
        end = start + len(batch_ids)
        if end == len(document_ids) or end % checkpoint_documents < document_batch_size:
            save_occurrence_checkpoint(
                checkpoint_path,
                state,
                next_document=end,
                selected_neurons=selected_neurons,
                signs=signs,
            )
            atomic_write_json(
                partial_output_path,
                occurrence_payload(
                    state=state,
                    selected_neurons=selected_neurons,
                    groups=groups,
                    signs=signs,
                    statistics=statistics,
                    metadata=metadata,
                    documents=documents,
                    tokenizer=tokenizer,
                    processed_documents=end,
                    total_documents=len(document_ids),
                    status="running",
                ),
            )
        log(f"Occurrence documents {end:,}/{len(document_ids):,}")
    atomic_write_json(
        output_path,
        occurrence_payload(
            state=state,
            selected_neurons=selected_neurons,
            groups=groups,
            signs=signs,
            statistics=statistics,
            metadata=metadata,
            documents=documents,
            tokenizer=tokenizer,
            processed_documents=len(document_ids),
            total_documents=len(document_ids),
            status="complete",
        ),
    )
    checkpoint_path.unlink(missing_ok=True)
    partial_output_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.document_batch_size <= 0 or args.checkpoint_documents <= 0:
        raise ValueError("batch and checkpoint sizes must be positive")
    if args.profile_tokens <= 0 or args.profile_tokens > EXPECTED_SAMPLE_COUNT:
        raise ValueError("--profile-tokens must be in 1..1,000,000")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default experiment")
    activations = args.activations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dataset = ActivationDataset(activations)
    validate_inputs(dataset, args.layer)
    provenance = dataset.manifest["provenance"]
    resolved = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "model": dataset.model,
        "activation_dataset": str(activations),
        "activation_manifest_sha256": sha256(activations / "activations.json"),
        "layer": args.layer,
        "layer_indexing": "transformer_blocks_zero_based",
        "quantity": "post-gating MLP coefficient entering down_proj",
        "direction": "corresponding column of W_downproj",
        "statistics_tokens": EXPECTED_SAMPLE_COUNT,
        "statistics_sampling": "all cached fitting rows",
        "profile_tokens": args.profile_tokens,
        "profile_sampling": "seeded uniform without replacement from fitting rows",
        "profile_seed": args.seed,
        "dominant_sign": "sign of population skewness",
        "selection": "lowest 3, seeded random 3 above median excluding extremes, highest 3",
        "top_k_occurrences": args.top_k,
        "source_dataset": provenance["dataset"],
        "candidate_tokens": provenance["candidate_tokens"],
        "context_length": provenance["context_length"],
        "document_framing": provenance["document_framing"],
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
    profiles_complete = validate_profiles(profiles_path, top_k=args.top_k)
    if profiles_complete and not statistics_complete:
        raise ValueError("profiles exist without a valid statistics dependency")
    completed = set()
    if statistics_complete:
        completed.add("statistics")
    if profiles_complete:
        completed.add("occurrences")
    if len(completed) == 2:
        log("All requested phases are already complete and compatible.")
        run.set_status("complete", complete=True)
        return
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · MLP neuron kurtosis profiles",
        completed=len(completed),
        total=2,
        completed_unit_ids=completed,
        unit_label="phases",
        source_dirty=bool(source_state.get("dirty")),
    ) as display:
        warn_if_dirty(source_state)
        display.phase("Loading pinned tokenizer and replaying cached documents", layer=args.layer)
        tokenizer = AutoTokenizer.from_pretrained(
            dataset.model["repo_id"],
            revision=dataset.model.get("revision"),
            trust_remote_code=True,
        )
        documents = load_documents(dataset, tokenizer)
        metadata = validate_documents(dataset, documents)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        profile_rows = torch.randperm(dataset.sample_count, generator=generator)[
            : args.profile_tokens
        ].sort().values
        all_rows = torch.arange(dataset.sample_count, dtype=torch.long)
        statistics_rows = rows_by_document(metadata, all_rows)
        occurrence_rows = rows_by_document(metadata, profile_rows)
        display.phase("Loading Qwen model", layer=args.layer)
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
        projection = down_projection(model, args.layer)
        if int(projection.weight.shape[0]) != EXPECTED_HIDDEN_SIZE:
            raise ValueError(f"unexpected down_proj shape: {tuple(projection.weight.shape)}")
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        if "statistics" not in completed:
            display.phase("Measuring population moments", layer=args.layer)
            run_statistics(
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
            display.complete_unit("statistics", refresh=True)
        else:
            log(f"Reusing completed neuron statistics: {statistics_path}")
        with np.load(statistics_path, allow_pickle=False) as stored:
            statistics = {key: stored[key].copy() for key in stored.files}
        selected_neurons, groups = select_neurons(
            statistics["excess_kurtosis"], seed=args.seed
        )
        if "occurrences" not in completed:
            display.phase("Finding dominant-tail occurrences", layer=args.layer)
            run_occurrences(
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
            display.complete_unit("occurrences", refresh=True)
        else:
            log(f"Reusing completed selected-neuron profiles: {profiles_path}")
        display.phase("Finalizing run")
    atomic_write_json(
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
