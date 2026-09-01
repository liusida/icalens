"""Run the suffix-sweep effective-receptive-field experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from icalens import ICALens, __version__
from icalens._capture import transformer_blocks
from icalens.cli._status import log

from ._display import ExperimentDisplay
from ._run import ResumableRun, atomic_write_json
from ._source_provenance import source_provenance, warn_if_dirty
from .erf_gradient import (
    _component_units,
    _lens_identity,
    _load_model,
    _parse_lenses,
    _prepare_component_inputs,
    _prepare_occurrence,
    _unit_id,
)

SCHEMA_VERSION = 1
FORMAT = "icalens.erf_suffix_sweep"
PREPARED_FORMAT = "icalens.erf_suffix_sweep.prepared_layer"
METHOD = "suffix-length-sweep-top15-selected-tail"


class _CapturedLayer(RuntimeError):
    """Stop a model forward after capturing the requested residual stream."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icalens experiment erf-suffix-sweep", description=__doc__
    )
    parser.add_argument(
        "--lens",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each local profiled ICA Lens.",
    )
    parser.add_argument("--components-per-layer", type=int, default=100)
    parser.add_argument("--occurrences-per-component", type=int, default=20)
    parser.add_argument("--max-suffix-length", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--batch-token-budget", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.components_per_layer,
            args.occurrences_per_component,
            args.max_suffix_length,
            args.top_k,
            args.max_batch_size,
            args.batch_token_budget,
        )
        < 1
    ):
        raise ValueError("component, occurrence, suffix, top-k, and batch values must be positive")

    lens_specs = _parse_lenses(args.lens)
    output = args.output.expanduser().resolve()
    cached = _load_prepared_run(output, lens_specs=lens_specs, args=args)
    lenses = {label: ICALens.from_pretrained(path) for label, path in lens_specs.items()}
    if cached is None:
        selections: dict[str, dict[str, list[int]]] = {}
        prepared_inputs: dict[str, dict[str, dict[str, Any]]] = {}
        for label, lens in lenses.items():
            selection, prepared = _prepare_component_inputs(
                lens,
                label=label,
                count=args.components_per_layer,
                seed=args.seed,
                occurrence_limit=args.occurrences_per_component,
            )
            selections[label] = selection
            prepared_inputs[label] = prepared
        resolved = _resolved_configuration(
            lens_specs=lens_specs,
            lenses=lenses,
            selections=selections,
            args=args,
        )
    else:
        resolved, selections, prepared_inputs = cached
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    source = source_provenance()
    run = ResumableRun.open(output=output, resolved=resolved, source=source, status="measuring")
    if cached is None:
        _write_prepared_inputs(output, prepared_inputs)
    units = _component_units(selections)
    completed_ids = {
        unit_id
        for unit_id, label, layer, component in units
        if _component_checkpoint_valid(
            _component_path(output, label, layer, component),
            label=label,
            layer=layer,
            component=component,
        )
    }
    with ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · suffix-sweep effective receptive field",
        completed=len(completed_ids),
        total=len(units),
        completed_unit_ids=completed_ids,
        unit_label="components",
        recent_label="Recent ERF output",
        detail_filename="erf-suffix-sweep-detail.log",
        source_dirty=bool(source.get("dirty")),
    ) as display:
        warn_if_dirty(source)
        log(
            f"Validated suffix-sweep ERF configuration: {len(lenses)} models, "
            f"{len(units)} components, up to {args.occurrences_per_component} occurrences each."
        )
        for label, lens in lenses.items():
            pending = [
                (int(layer), component)
                for layer, components in sorted(
                    selections[label].items(), key=lambda item: int(item[0])
                )
                for component in components
                if _unit_id(label, int(layer), component) not in completed_ids
            ]
            if not pending:
                log(f"{label}: all selected components already complete; skipping model load.")
                continue
            display.phase("Loading model", model=label)
            log(f"Loading {label}: {lens.model_id}@{lens.model_revision}.")
            model = _load_model(lens, device=args.device)
            tokenizer = AutoTokenizer.from_pretrained(
                lens.model_id,
                revision=lens.model_revision,
                use_fast=True,
                trust_remote_code=True,
            )
            datasets: dict[str, Any] = {}
            token_cache: dict[tuple[str, int, int], list[int]] = {}
            try:
                pending_by_layer: dict[int, list[int]] = {}
                for layer, component in pending:
                    pending_by_layer.setdefault(layer, []).append(component)
                for layer, components in pending_by_layer.items():
                    display.phase("Sweeping suffixes", model=label, layer=layer)

                    def checkpoint(
                        component: int,
                        result: dict[str, Any],
                        *,
                        _label: str = label,
                        _layer: int = layer,
                    ) -> None:
                        result.update(model_label=_label, method=METHOD)
                        atomic_write_json(
                            _component_path(output, _label, _layer, component), result
                        )
                        display.complete_unit(_unit_id(_label, _layer, component), refresh=True)
                        log(
                            f"Checkpointed {_label} layer {_layer} C{component}: "
                            f"mean ERF {result['suffix_erf_mean']:.4f}, "
                            f"recovered {result['n_recovered']}/{result['n_occurrences']}."
                        )

                    _measure_layer(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        datasets=datasets,
                        token_cache=token_cache,
                        layer=layer,
                        prepared_components={
                            component: prepared_inputs[label][str(layer)][str(component)]
                            for component in components
                        },
                        max_suffix_length=args.max_suffix_length,
                        top_k=args.top_k,
                        max_batch_size=args.max_batch_size,
                        batch_token_budget=args.batch_token_budget,
                        device=args.device,
                        checkpoint=checkpoint,
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        display.phase("Summarizing")
        _write_summaries(output, units)
        run.set_status("complete", complete=True)
        log(f"Suffix-sweep ERF experiment complete: {output}")


def _measure_layer(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    datasets: dict[str, Any],
    token_cache: dict[tuple[str, int, int], list[int]],
    layer: int,
    prepared_components: dict[int, dict[str, Any]],
    max_suffix_length: int,
    top_k: int,
    max_batch_size: int,
    batch_token_budget: int,
    device: str,
    checkpoint: Any,
) -> None:
    """Sweep a layer together so short suffixes can use large cross-component batches."""
    provenance = lens.metadata["layers"][str(layer)]["fitting"]["provenance"]
    states: dict[int, dict[str, Any]] = {}
    for component, value in prepared_components.items():
        prepared = [
            _prepare_occurrence(
                occurrence,
                occurrence_rank=rank,
                provenance=provenance,
                tokenizer=tokenizer,
                datasets=datasets,
                token_cache=token_cache,
            )
            for rank, occurrence in enumerate(value["occurrences"], start=1)
        ]
        states[component] = {
            "direction": str(value["tail_direction"]),
            "prepared": prepared,
            "unresolved": {int(item["occurrence_rank"]): item for item in prepared},
            "observations": {rank: [] for rank in range(1, len(prepared) + 1)},
        }
    block = transformer_blocks(model)[layer]
    completed: set[int] = set()
    for suffix_length in range(1, max_suffix_length + 1):
        active = [
            (component, state["direction"], item)
            for component, state in states.items()
            if component not in completed
            for item in state["unresolved"].values()
            if len(item["content_ids"]) >= suffix_length
        ]
        batch_size = _adaptive_batch_size(
            suffix_length,
            max_batch_size=max_batch_size,
            token_budget=batch_token_budget,
        )
        for start in range(0, len(active), batch_size):
            batch = active[start : start + batch_size]
            measured = _measure_mixed_batch(
                lens=lens,
                model=model,
                block=block,
                tokenizer=tokenizer,
                layer=layer,
                batch=batch,
                suffix_length=suffix_length,
                top_k=top_k,
                framing=provenance["document_framing"],
                device=device,
            )
            for component, value in measured:
                rank = int(value["occurrence_rank"])
                states[component]["observations"][rank].append(value)
                if value["recovered"]:
                    states[component]["unresolved"].pop(rank, None)
        for component, state in states.items():
            if component not in completed and not state["unresolved"]:
                checkpoint(
                    component,
                    _finish_component_result(
                        layer=layer,
                        component=component,
                        direction=state["direction"],
                        prepared=state["prepared"],
                        observations=state["observations"],
                        max_suffix_length=max_suffix_length,
                        top_k=top_k,
                    ),
                )
                completed.add(component)
        if len(completed) == len(states):
            return
    for component, state in states.items():
        if component not in completed:
            checkpoint(
                component,
                _finish_component_result(
                    layer=layer,
                    component=component,
                    direction=state["direction"],
                    prepared=state["prepared"],
                    observations=state["observations"],
                    max_suffix_length=max_suffix_length,
                    top_k=top_k,
                ),
            )


def _adaptive_batch_size(suffix_length: int, *, max_batch_size: int, token_budget: int) -> int:
    """Keep approximately the same number of lexical tokens in each forward batch."""
    if min(suffix_length, max_batch_size, token_budget) < 1:
        raise ValueError("suffix length, maximum batch size, and token budget must be positive")
    return max(1, min(max_batch_size, token_budget // suffix_length))


def _measure_mixed_batch(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    block: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    batch: list[tuple[int, str, dict[str, Any]]],
    suffix_length: int,
    top_k: int,
    framing: dict[str, Any],
    device: str,
) -> list[tuple[int, dict[str, Any]]]:
    prepared = [item for _component, _direction, item in batch]
    sequences = []
    for item in prepared:
        ids = list(item["content_ids"][-suffix_length:])
        if item["framing_offset"]:
            ids.insert(0, int(framing["token_id"]))
        sequences.append(ids)
    scores = _suffix_scores(
        lens=lens,
        model=model,
        block=block,
        tokenizer=tokenizer,
        layer=layer,
        sequences=sequences,
        device=device,
    )
    components = torch.tensor([component for component, _direction, _item in batch], device=device)
    rows = torch.arange(len(batch), device=device)
    selected = scores[rows, components]
    ranks = (scores.abs() > selected.abs().unsqueeze(1)).sum(1) + 1
    positive = torch.tensor(
        [direction == "positive" for _component, direction, _item in batch], device=device
    )
    sign_ok = torch.where(positive, selected > 0, selected < 0)
    recovered = (ranks <= top_k) & sign_ok
    return [
        (
            component,
            {
                "occurrence_rank": int(item["occurrence_rank"]),
                "suffix_length": suffix_length,
                "score": float(selected[row]),
                "absolute_score_rank": int(ranks[row]),
                "sign_matches_selected_tail": bool(sign_ok[row]),
                "recovered": bool(recovered[row]),
            },
        )
        for row, (component, _direction, item) in enumerate(batch)
    ]


def _suffix_scores(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    block: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    sequences: list[list[int]],
    device: str,
) -> torch.Tensor:
    lengths = [len(ids) for ids in sequences]
    maximum = max(lengths)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    input_ids = torch.full((len(sequences), maximum), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
    captured: dict[str, torch.Tensor] = {}

    def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        rows = torch.arange(len(sequences), device=hidden.device)
        positions = torch.tensor(lengths, device=hidden.device) - 1
        captured["target"] = hidden[rows, positions].float()
        raise _CapturedLayer

    handle = block.register_forward_hook(hook)
    try:
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _CapturedLayer:
            pass
    finally:
        handle.remove()
    if "target" not in captured:
        raise RuntimeError("requested residual stream was not captured")
    return lens.transform(captured["target"], layer=layer).detach()


def _finish_component_result(
    *,
    layer: int,
    component: int,
    direction: str,
    prepared: list[dict[str, Any]],
    observations: dict[int, list[dict[str, Any]]],
    max_suffix_length: int,
    top_k: int,
) -> dict[str, Any]:
    sentinel = max_suffix_length + 1
    results = []
    for item in prepared:
        rank = int(item["occurrence_rank"])
        trace = observations[rank]
        recovered = next((value for value in trace if value["recovered"]), None)
        occurrence = item["occurrence"]
        results.append(
            {
                "occurrence_rank": rank,
                "source_index": int(occurrence["source_index"]),
                "position": int(occurrence["position"]),
                "token_id": int(occurrence["token_id"]),
                "text": str(occurrence["text"]),
                "context": str(occurrence["context"]),
                "stored_score": float(occurrence["score"]),
                "stored_energy": float(occurrence["energy"]),
                "content_tokens": len(item["content_ids"]),
                "recovered": recovered is not None,
                "suffix_erf": (
                    int(recovered["suffix_length"]) if recovered is not None else sentinel
                ),
                "suffix_erf_label": (
                    str(recovered["suffix_length"])
                    if recovered is not None
                    else f">{max_suffix_length}"
                ),
                "sweep": trace,
            }
        )
    values = [int(item["suffix_erf"]) for item in results]
    recovered_count = sum(bool(item["recovered"]) for item in results)
    return {
        "layer": layer,
        "component": component,
        "tail_direction": direction,
        "max_suffix_length": max_suffix_length,
        "unrecovered_sentinel": sentinel,
        "top_k": top_k,
        "n_occurrences": len(results),
        "n_recovered": recovered_count,
        "recovered_fraction": recovered_count / len(results),
        "suffix_erf_median": statistics.median(values),
        "suffix_erf_mean": statistics.mean(values),
        "suffix_erf_min": min(values),
        "suffix_erf_max": max(values),
        "occurrences": results,
    }


def _recovery_from_scores(
    scores: torch.Tensor, *, component: int, direction: str, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute-score rank, selected-tail match, and recovery."""
    if scores.ndim != 2 or not 0 <= component < scores.shape[1] or top_k < 1:
        raise ValueError("scores, component, and top_k are incompatible")
    if direction not in ("positive", "negative"):
        raise ValueError("direction must be positive or negative")
    selected = scores[:, component]
    ranks = (scores.abs() > selected.abs().unsqueeze(1)).sum(1) + 1
    sign_ok = selected > 0 if direction == "positive" else selected < 0
    return ranks, sign_ok, (ranks <= top_k) & sign_ok


def _resolved_configuration(
    *,
    lens_specs: dict[str, Path],
    lenses: dict[str, ICALens],
    selections: dict[str, dict[str, list[int]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "icalens_version": __version__,
        "method": METHOD,
        "recovery": "absolute-score rank <= top_k and score on selected tail",
        "occurrence_erf": "first recovered suffix length; max_suffix_length + 1 means censored",
        "component_erf": "mean occurrence ERF with censored sentinel",
        "components_per_layer": args.components_per_layer,
        "occurrences_per_component": args.occurrences_per_component,
        "max_suffix_length": args.max_suffix_length,
        "top_k": args.top_k,
        "seed": args.seed,
        "lens_order": list(lens_specs),
        "lenses": {
            label: _lens_identity(path, lenses[label], selections[label])
            for label, path in lens_specs.items()
        },
        "selections": selections,
    }


def _prepared_layer_path(output: Path, label: str, layer: int) -> Path:
    return output / "prepared" / label / f"layer_{layer:02d}.json"


def _write_prepared_inputs(
    output: Path, prepared_inputs: dict[str, dict[str, dict[str, Any]]]
) -> None:
    for label, layers in prepared_inputs.items():
        for layer_text, components in layers.items():
            atomic_write_json(
                _prepared_layer_path(output, label, int(layer_text)),
                {
                    "format": PREPARED_FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "model_label": label,
                    "layer": int(layer_text),
                    "components": components,
                },
            )


def _load_prepared_run(
    output: Path, *, lens_specs: dict[str, Path], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    run_path = output / "run.json"
    if not run_path.is_file():
        return None
    try:
        resolved = json.loads(run_path.read_text(encoding="utf-8"))["resolved"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{run_path} has no valid resolved configuration") from error
    keys = (
        "format",
        "schema_version",
        "method",
        "components_per_layer",
        "occurrences_per_component",
        "max_suffix_length",
        "top_k",
        "seed",
        "lens_order",
    )
    requested = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "components_per_layer": args.components_per_layer,
        "occurrences_per_component": args.occurrences_per_component,
        "max_suffix_length": args.max_suffix_length,
        "top_k": args.top_k,
        "seed": args.seed,
        "lens_order": list(lens_specs),
    }
    existing = {key: resolved.get(key) for key in keys}
    if existing != requested:
        differences = [
            f"{key}: {existing[key]!r} -> {requested[key]!r}"
            for key in keys
            if existing[key] != requested[key]
        ]
        raise ValueError(
            f"{run_path} belongs to a different configuration "
            f"({'; '.join(differences)}); choose another output"
        )
    for label, path in lens_specs.items():
        identity = resolved.get("lenses", {}).get(label, {})
        if identity.get("path") != str(path):
            raise ValueError(f"{run_path} uses a different Lens path for {label}")
        current = _lens_identity(path, ICALens.from_pretrained(path), resolved["selections"][label])
        if current["manifest_sha256"] != identity.get("manifest_sha256") or current[
            "profile_sha256"
        ] != identity.get("profile_sha256"):
            raise ValueError(
                f"prepared suffix-sweep inputs are stale because {label} changed; "
                "choose another output"
            )
    selections = resolved.get("selections")
    if not isinstance(selections, dict):
        raise ValueError(f"{run_path} has no valid component selections")
    prepared: dict[str, Any] = {}
    for label, layers in selections.items():
        prepared[label] = {}
        for layer_text, selected in layers.items():
            path = _prepared_layer_path(output, label, int(layer_text))
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            components = payload.get("components")
            if (
                payload.get("format") != PREPARED_FORMAT
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("model_label") != label
                or payload.get("layer") != int(layer_text)
                or not isinstance(components, dict)
                or sorted(int(value) for value in components) != sorted(selected)
            ):
                return None
            prepared[label][layer_text] = components
    return resolved, selections, prepared


def _component_path(output: Path, label: str, layer: int, component: int) -> Path:
    return output / "components" / label / f"layer_{layer:02d}" / f"C{component:05d}.json"


def _component_checkpoint_valid(path: Path, *, label: str, layer: int, component: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("model_label") == label
        and value.get("layer") == layer
        and value.get("component") == component
        and value.get("method") == METHOD
        and isinstance(value.get("suffix_erf_median"), (int, float))
        and isinstance(value.get("occurrences"), list)
        and len(value["occurrences"]) == value.get("n_occurrences")
    )


def _write_summaries(output: Path, units: list[tuple[str, str, int, int]]) -> None:
    rows = []
    for _, label, layer, component in units:
        value = json.loads(_component_path(output, label, layer, component).read_text())
        rows.append(
            {
                "model": label,
                "layer": layer,
                "component": component,
                "suffix_erf_median": value["suffix_erf_median"],
                "suffix_erf_mean": value["suffix_erf_mean"],
                "n_occurrences": value["n_occurrences"],
                "n_recovered": value["n_recovered"],
                "recovered_fraction": value["recovered_fraction"],
            }
        )
    atomic_write_json(output / "summary.json", {"format": FORMAT, "rows": rows})
    path = output / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
