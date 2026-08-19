"""Evaluate top-k dictionary reconstruction on held-out text."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Iterable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens, __version__
from icalens._capture import transformer_blocks
from icalens.analysis import _document_framing_for_layer

from ._saebench_worker import SAEFeatureEncoder, _BenchmarkDisplay
from ._source_provenance import source_provenance, warn_if_dirty
from .saebench_sparse_probing import _prepare_layer_baselines, _resolve_baselines

SAE_TRAINING_CONTEXT_CONTROLS = {"openai-community/gpt2": 64}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment reconstruction", description=__doc__
    )
    parser.add_argument("--lens", required=True, help="Local ICA Lens path or Hugging Face repo.")
    parser.add_argument("--layers", required=True, help="Comma-separated layers, or 'all'.")
    parser.add_argument(
        "--preset", choices=("smoke", "paper", "pile10k"), default="smoke"
    )
    parser.add_argument(
        "--baselines", default="all", help="Comparison baselines: sae,pca,random,all."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-layers-at-once",
        default="all",
        help="Layers captured in each shared model pass, or 'all' (default: all).",
    )
    parser.add_argument("--max-tokens-per-dataset", type=int, default=None)
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override the preset's maximum tokenized document length.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = source_provenance()
    warn_if_dirty(source)
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layers, lens.available_layers)
    settings = _load_preset(args.preset)
    if args.max_tokens_per_dataset is not None:
        if args.max_tokens_per_dataset < 1:
            raise ValueError("--max-tokens-per-dataset must be positive")
        settings["max_tokens_per_dataset"] = int(args.max_tokens_per_dataset)
    if args.context_length is not None:
        if args.context_length < 1:
            raise ValueError("--context-length must be positive")
        settings["context_length"] = int(args.context_length)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        settings["batch_size"] = int(args.batch_size)
    capture_layers_at_once = _parse_capture_layers_at_once(
        args.capture_layers_at_once, selected_layer_count=len(layers)
    )
    baselines = _resolve_baselines(lens.model_id, args.baselines)
    resolved = {
        "experiment": "reconstruction",
        "experiment_schema_version": 3,
        "icalens_version": __version__,
        "lens": str(args.lens),
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "activation_site": lens.activation_site,
        "row_normalize": lens.row_normalize,
        "layers": layers,
        "preset": settings,
        "baselines": baselines,
        "capture_layers_at_once": capture_layers_at_once,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if lens.activation_site != "resid_post":
        raise ValueError("reconstruction currently supports activation_site='resid_post'")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    run = _load_run(run_path, resolved)
    run["icalens_source"] = source
    _write_json(run_path, run)
    pending = [layer for layer in layers if not _layer_path(output, layer).is_file()]
    if not pending:
        _finish(output, run_path, run, resolved, layers)
        print(f"All requested layers are already complete: {output}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        lens.model_id, revision=lens.model_revision, use_fast=True
    )
    # Document framing must immediately precede content, so batched evaluation
    # uses right padding regardless of the tokenizer's generation default.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(lens, device=args.device)
    try:
        _run_dataset_first(
            lens=lens,
            model=model,
            tokenizer=tokenizer,
            layers=pending,
            settings=settings,
            baselines=baselines,
            capture_layers_at_once=capture_layers_at_once,
            output=output,
            resolved=resolved,
            run=run,
            run_path=run_path,
            device=args.device,
        )
    except Exception as error:
        for layer in pending:
            if not _layer_path(output, layer).is_file():
                run["layer_runs"][str(layer)] = {"status": "failed", "error": str(error)}
        _write_json(run_path, run)
        raise
    _finish(output, run_path, run, resolved, layers)
    print(f"Experiment complete: {output}")
    print(f"Create figures with: icalens experiment figure reconstruction {output}")


def _load_model(lens: ICALens, *, device: str) -> torch.nn.Module:
    if device.startswith("cuda"):
        model = load_model_to_cuda(
            AutoModelForCausalLM,
            lens.model_id,
            revision=lens.model_revision,
            device=device,
            dtype=torch.bfloat16,
            touch="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lens.model_id, revision=lens.model_revision
        ).to(device)
    model.eval()
    return model


def _run_dataset_first(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    layers: list[int],
    settings: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    capture_layers_at_once: int,
    output: Path,
    resolved: dict[str, Any],
    run: dict[str, Any],
    run_path: Path,
    device: str,
) -> None:
    datasets = list(settings["datasets"])
    selected_layers = [int(layer) for layer in resolved["layers"]]
    completed = len(datasets) * sum(
        _layer_path(output, layer).is_file() for layer in selected_layers
    )
    completed += sum(
        _dataset_result_path(output, layer, index).is_file()
        for layer in layers
        for index in range(len(datasets))
    )
    total = len(datasets) * len(selected_layers)
    started = time.time()
    display = _BenchmarkDisplay(
        output=output / "logs",
        completed=completed,
        total=total,
        run_initial=completed,
        run_started_at=started,
        title="ICA Lens · held-out reconstruction",
        item_label="Layer",
        items_label="Layers",
        recent_label="Recent reconstruction output",
        detail_filename="reconstruction-detail.log",
    )
    with display:
        for dataset_index, dataset_config in enumerate(datasets):
            missing = [
                layer
                for layer in layers
                if not _dataset_result_path(output, layer, dataset_index).is_file()
            ]
            if not missing:
                _remove_activation_cache(_dataset_cache_dir(output, dataset_index))
                continue
            labels = [f"L{layer}" for layer in layers]
            completed_labels = {
                f"L{layer}"
                for layer in layers
                if _dataset_result_path(output, layer, dataset_index).is_file()
            }
            display.set_dataset(
                str(dataset_config.get("domain", dataset_config["repo_id"])),
                index=dataset_index + 1,
                total=len(datasets),
                methods=labels,
                completed_methods=completed_labels,
            )
            groups = [
                missing[index : index + capture_layers_at_once]
                for index in range(0, len(missing), capture_layers_at_once)
            ]
            for group in groups:
                cache_dir = _dataset_cache_dir(output, dataset_index)
                uncached = [
                    layer for layer in group if not _activation_path(cache_dir, layer).is_file()
                ]
                if uncached:
                    display.set_phase(
                        "shared capture " + ",".join(f"L{layer}" for layer in uncached)
                    )
                    activations, positions = _capture_dataset(
                        lens=lens,
                        model=model,
                        tokenizer=tokenizer,
                        texts=_dataset_texts(dataset_config),
                        layers=uncached,
                        context_length=int(settings["context_length"]),
                        batch_size=int(settings["batch_size"]),
                        max_tokens=int(settings["max_tokens_per_dataset"]),
                        device=device,
                    )
                    _save_activation_cache(cache_dir, activations, positions)
                    del activations, positions
                for _, layer in display.track_methods(
                    [(f"L{layer}", layer) for layer in group]
                ):
                    activation_file = _activation_path(cache_dir, layer)
                    tensors = load_file(activation_file)
                    result = _evaluate_layer(
                        lens=lens,
                        layer=layer,
                        activations=tensors["activations"],
                        positions=tensors["positions"],
                        k_values=[int(value) for value in settings["k_values"]],
                        baselines=baselines,
                        device=device,
                    )
                    result["dataset"] = dataset_config
                    _write_json(_dataset_result_path(output, layer, dataset_index), result)
                    activation_file.unlink(missing_ok=True)
                    del tensors, result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            _remove_activation_cache(_dataset_cache_dir(output, dataset_index))
            for layer in layers:
                _finalize_layer_if_complete(
                    output=output,
                    layer=layer,
                    datasets=datasets,
                    resolved=resolved,
                    run=run,
                    run_path=run_path,
                )


def _dataset_result_path(output: Path, layer: int, dataset_index: int) -> Path:
    return output / "checkpoints" / "datasets" / f"dataset_{dataset_index:02d}" / f"layer_{layer:02d}.json"


def _dataset_cache_dir(output: Path, dataset_index: int) -> Path:
    return output / "checkpoints" / "reconstruction-activations" / f"dataset_{dataset_index:02d}"


def _activation_path(cache_dir: Path, layer: int) -> Path:
    return cache_dir / f"layer_{layer:02d}.safetensors"


def _save_activation_cache(
    cache_dir: Path,
    activations: dict[int, torch.Tensor],
    positions: torch.Tensor,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for layer, values in activations.items():
        destination = _activation_path(cache_dir, layer)
        temporary = destination.with_suffix(".safetensors.partial")
        save_file(
            {"activations": values.contiguous(), "positions": positions.contiguous()},
            temporary,
        )
        os.replace(temporary, destination)


def _remove_activation_cache(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _finalize_layer_if_complete(
    *,
    output: Path,
    layer: int,
    datasets: list[dict[str, Any]],
    resolved: dict[str, Any],
    run: dict[str, Any],
    run_path: Path,
) -> bool:
    paths = [_dataset_result_path(output, layer, index) for index in range(len(datasets))]
    if not all(path.is_file() for path in paths):
        return False
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    payload = {
        "experiment": resolved,
        "layer": layer,
        "datasets": metrics,
        "rows": _aggregate_layer(layer, metrics),
    }
    _write_json(_layer_path(output, layer), payload)
    run["layer_runs"][str(layer)] = {"status": "complete"}
    _write_json(run_path, run)
    return True


def _dataset_texts(config: dict[str, Any]) -> Iterable[str]:
    data_files = config.get("data_files")
    if config.get("loader") == "parquet" and isinstance(data_files, str) and data_files.startswith(
        "hf://datasets/"
    ):
        # datasets' remote Parquet streaming reader can leave its background
        # range-fetch workers alive after an early stop. Materialize this one
        # pinned shard through huggingface_hub, then iterate it locally so the
        # experiment process exits promptly and predictably.
        repo_id = str(config["repo_id"])
        prefix = f"hf://datasets/{repo_id}/"
        if not data_files.startswith(prefix):
            raise ValueError(f"Parquet URL does not belong to {repo_id}: {data_files}")
        local = hf_hub_download(
            repo_id=repo_id,
            filename=data_files.removeprefix(prefix),
            repo_type="dataset",
        )
        import pyarrow.parquet as pq

        column = str(config.get("text_column", "text"))
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(columns=[column], batch_size=256):
            for value in batch.column(0).to_pylist():
                text = _dataset_value_to_text(value, config)
                if text is not None:
                    yield text
        return
    kwargs: dict[str, Any] = {
        "path": str(config.get("loader", config["repo_id"])),
        "split": str(config.get("split", "train")),
        "streaming": True,
    }
    if config.get("config") is not None:
        kwargs["name"] = str(config["config"])
    if config.get("data_files") is not None:
        kwargs["data_files"] = config["data_files"]
    dataset = load_dataset(**kwargs)
    column = str(config.get("text_column", "text"))
    for row in dataset:
        text = _dataset_value_to_text(row.get(column), config)
        if text is not None:
            yield text


def _dataset_value_to_text(value: Any, config: dict[str, Any]) -> str | None:
    if config.get("format") == "conversation":
        if not isinstance(value, list):
            return None
        lines: list[str] = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                lines.append(f"{role.strip().capitalize()}: {content.strip()}")
        rendered = "\n".join(lines)
        return rendered if rendered else None
    return value if isinstance(value, str) and value.strip() else None


def _capture_dataset(
    *,
    lens: ICALens,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Iterable[str],
    layers: list[int],
    context_length: int,
    batch_size: int,
    max_tokens: int,
    device: str,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    framing = _document_framing_for_layer(lens, layers[0])
    prefix_id = framing.get("token_id") if framing.get("strategy") != "none" else None
    content_length = context_length - int(prefix_id is not None)
    buffers: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    position_parts: list[torch.Tensor] = []
    total = 0
    batch: list[str] = []
    progress = tqdm(total=max_tokens, desc="Capture held-out activations", unit="tok")
    for text in texts:
        batch.append(text)
        if len(batch) < batch_size:
            continue
        captured, positions, count = _capture_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch,
            layers=layers,
            prefix_id=prefix_id,
            content_length=content_length,
            remaining=max_tokens - total,
            device=device,
        )
        for layer in layers:
            buffers[layer].append(captured[layer].cpu())
        position_parts.append(positions)
        total += count
        progress.update(count)
        batch.clear()
        if total >= max_tokens:
            break
    if batch and total < max_tokens:
        captured, positions, count = _capture_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch,
            layers=layers,
            prefix_id=prefix_id,
            content_length=content_length,
            remaining=max_tokens - total,
            device=device,
        )
        for layer in layers:
            buffers[layer].append(captured[layer].cpu())
        position_parts.append(positions)
        total += count
        progress.update(count)
    progress.close()
    if total == 0:
        raise RuntimeError("held-out dataset yielded no usable tokens")
    return (
        {layer: torch.cat(buffers[layer], dim=0) for layer in layers},
        torch.cat(position_parts),
    )


def _capture_batch(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    prefix_id: int | None,
    content_length: int,
    remaining: int,
    device: str,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=content_length,
        padding=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"]
    mask = encoded["attention_mask"].bool()
    if prefix_id is not None:
        prefix = torch.full((ids.shape[0], 1), int(prefix_id), dtype=ids.dtype)
        ids = torch.cat((prefix, ids), dim=1)
        mask = torch.cat((torch.zeros((ids.shape[0], 1), dtype=torch.bool), mask), dim=1)
    valid = mask.nonzero(as_tuple=False)
    if valid.shape[0] > remaining:
        valid = valid[:remaining]
    positions = valid[:, 1].cpu() - int(prefix_id is not None)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int) -> Any:
        def hook(_: Any, __: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            batch_indices = valid[:, 0].to(hidden.device)
            token_indices = valid[:, 1].to(hidden.device)
            # Preserve the model activation dtype in the shared CPU cache. The
            # evaluator promotes one layer at a time to float32, so storing bf16
            # activations from bf16 models avoids a needless 2x cache/RAM cost.
            captured[layer] = hidden[batch_indices, token_indices].detach()
        return hook

    blocks = transformer_blocks(model)
    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            model(
                input_ids=ids.to(device),
                attention_mask=torch.cat(
                    (torch.ones((mask.shape[0], 1), dtype=torch.bool), mask[:, 1:]), dim=1
                ).to(device) if prefix_id is not None else mask.to(device),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured, positions, int(valid.shape[0])


def _evaluate_layer(
    *,
    lens: ICALens,
    layer: int,
    activations: torch.Tensor,
    positions: torch.Tensor,
    k_values: list[int],
    baselines: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    artifact = lens._get_layer(layer)
    assert artifact.center is not None
    assert artifact.reading_matrix is not None
    assert artifact.writing_matrix is not None
    target = activations.to(device=device, dtype=torch.float32)
    norm = target.norm(dim=-1, keepdim=True)
    work = F.normalize(target, dim=-1, eps=lens.norm_eps) if lens.row_normalize else target
    center = torch.as_tensor(artifact.center, device=device, dtype=torch.float32)
    reading = torch.as_tensor(artifact.reading_matrix, device=device, dtype=torch.float32)
    writing = torch.as_tensor(artifact.writing_matrix, device=device, dtype=torch.float32)
    methods: dict[str, dict[str, Any]] = {}
    methods["ica"] = _linear_dictionary_metrics_batched(
        target, work, norm, center, reading, writing.T, k_values, restore_norm=lens.row_normalize
    )
    if "pca" in baselines:
        covariance = writing.double() @ writing.double().T
        _, vectors = torch.linalg.eigh(covariance)
        basis = vectors.flip(1).T.float()
        methods["pca"] = _linear_dictionary_metrics_batched(
            target, work, norm, center, basis, basis, k_values, restore_norm=lens.row_normalize
        )
    if "random" in baselines:
        generator = torch.Generator(device=device).manual_seed(
            int(baselines["random"].get("seed", 0)) + layer
        )
        matrix = torch.randn(
            (target.shape[-1], target.shape[-1]), generator=generator, device=device
        )
        basis = torch.linalg.qr(matrix).Q.T
        methods["random"] = _linear_dictionary_metrics_batched(
            target, work, norm, center, basis, basis, k_values, restore_norm=lens.row_normalize
        )
    if "sae" in baselines:
        prepared = _prepare_layer_baselines({"sae": baselines["sae"]}, layer=layer)
        snapshot = {
            "hidden_size": int(target.shape[-1]),
            "layer": layer,
            "saebench_model_name": lens.model_id,
            "baselines": prepared,
        }
        sae = SAEFeatureEncoder(snapshot, device=device, dtype=torch.float32)
        methods["sae"] = _sae_metrics_batched(target, sae, k_values)
        control_context = SAE_TRAINING_CONTEXT_CONTROLS.get(lens.model_id)
        if control_context is not None:
            control_mask = positions < control_context
            if control_mask.any():
                methods[f"sae_context_{control_context}"] = _sae_metrics_batched(
                    target[control_mask.to(target.device)], sae, k_values
                )
                methods[f"sae_context_{control_context}"]["position_control"] = {
                    "maximum_position_exclusive": control_context,
                    "n_tokens": int(control_mask.sum().item()),
                }
        del sae
    del target, work, norm, center, reading, writing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"n_tokens": int(activations.shape[0]), "methods": methods}


def _linear_dictionary_metrics(
    target: torch.Tensor,
    work: torch.Tensor,
    norm: torch.Tensor,
    center: torch.Tensor,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
    k_values: list[int],
    *,
    restore_norm: bool,
) -> dict[str, Any]:
    scores = (work - center) @ encoder.T
    contribution = scores.abs() * decoder.norm(dim=-1)
    curves: dict[str, Any] = {}
    for k in sorted(set([*k_values, int(scores.shape[-1])])):
        actual = min(k, int(scores.shape[-1]))
        indices = contribution.topk(actual, dim=-1).indices
        selected = torch.zeros_like(scores).scatter(1, indices, scores.gather(1, indices))
        reconstructed = selected @ decoder + center
        if restore_norm:
            reconstructed = F.normalize(reconstructed, dim=-1) * norm
        curves[str(k)] = _metrics(target, reconstructed)
    return {"curve": curves, "full_k": int(scores.shape[-1])}


def _linear_dictionary_metrics_batched(
    target: torch.Tensor,
    work: torch.Tensor,
    norm: torch.Tensor,
    center: torch.Tensor,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
    k_values: list[int],
    *,
    restore_norm: bool,
    batch_size: int = 2048,
) -> dict[str, Any]:
    collected: dict[str, dict[str, list[torch.Tensor]]] = {}
    for start in range(0, target.shape[0], batch_size):
        end = min(start + batch_size, target.shape[0])
        scores = (work[start:end] - center) @ encoder.T
        contribution = scores.abs() * decoder.norm(dim=-1)
        for k in sorted(set([*k_values, int(scores.shape[-1])])):
            actual = min(k, int(scores.shape[-1]))
            indices = contribution.topk(actual, dim=-1).indices
            selected = torch.zeros_like(scores).scatter(1, indices, scores.gather(1, indices))
            reconstructed = selected @ decoder + center
            if restore_norm:
                reconstructed = F.normalize(reconstructed, dim=-1) * norm[start:end]
            error, cosine = _metric_values(target[start:end], reconstructed)
            entry = collected.setdefault(str(k), {"nmse": [], "cosine": []})
            entry["nmse"].append(error.cpu())
            entry["cosine"].append(cosine.cpu())
    return {
        "curve": {
            key: {
                metric: _summary(torch.cat(parts))
                for metric, parts in values.items()
            }
            for key, values in collected.items()
        },
        "full_k": int(encoder.shape[0]),
    }


def _sae_metrics(
    target: torch.Tensor,
    codes: torch.Tensor,
    sae: SAEFeatureEncoder,
    k_values: list[int],
) -> dict[str, Any]:
    contribution = codes.abs() * sae.W_dec.norm(dim=-1)
    active = (codes != 0).sum(dim=-1).float()
    effective_k: dict[str, float] = {}
    curves: dict[str, Any] = {}
    for k in k_values:
        actual = min(k, int(codes.shape[-1]))
        indices = contribution.topk(actual, dim=-1).indices
        selected = torch.zeros_like(codes).scatter(1, indices, codes.gather(1, indices))
        effective_k[str(k)] = float((selected != 0).sum(dim=-1).float().mean().item())
        curves[str(k)] = _metrics(target, sae.decode(selected, reference=target))
    curves["native"] = _metrics(target, sae.decode(codes, reference=target))
    return {
        "curve": curves,
        "effective_k": effective_k,
        "native_mean_active": float(active.mean().item()),
        "width": int(codes.shape[-1]),
    }


def _sae_metrics_batched(
    target: torch.Tensor,
    sae: SAEFeatureEncoder,
    k_values: list[int],
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    collected: dict[str, dict[str, list[torch.Tensor]]] = {}
    active_parts: list[torch.Tensor] = []
    effective_parts: dict[str, list[torch.Tensor]] = {}
    decoder_norms = sae.W_dec.norm(dim=-1)
    for start in range(0, target.shape[0], batch_size):
        chunk = target[start : start + batch_size]
        with torch.inference_mode():
            codes = sae.encode(chunk)
        contribution = codes.abs() * decoder_norms
        active_parts.append((codes != 0).sum(dim=-1).float().cpu())
        for k in k_values:
            actual = min(k, int(codes.shape[-1]))
            indices = contribution.topk(actual, dim=-1).indices
            selected = torch.zeros_like(codes).scatter(1, indices, codes.gather(1, indices))
            effective_parts.setdefault(str(k), []).append(
                (selected != 0).sum(dim=-1).float().cpu()
            )
            error, cosine = _metric_values(
                chunk, sae.decode(selected, reference=chunk)
            )
            entry = collected.setdefault(str(k), {"nmse": [], "cosine": []})
            entry["nmse"].append(error.cpu())
            entry["cosine"].append(cosine.cpu())
        error, cosine = _metric_values(chunk, sae.decode(codes, reference=chunk))
        native = collected.setdefault("native", {"nmse": [], "cosine": []})
        native["nmse"].append(error.cpu())
        native["cosine"].append(cosine.cpu())
    return {
        "curve": {
            key: {
                metric: _summary(torch.cat(parts))
                for metric, parts in values.items()
            }
            for key, values in collected.items()
        },
        "effective_k": {
            key: float(torch.cat(parts).mean().item())
            for key, parts in effective_parts.items()
        },
        "native_mean_active": float(torch.cat(active_parts).mean().item()),
        "width": int(sae.W_dec.shape[0]),
    }


def _metrics(target: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, Any]:
    error, cosine = _metric_values(target, reconstructed)
    return {"nmse": _summary(error), "cosine": _summary(cosine)}


def _metric_values(
    target: torch.Tensor, reconstructed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    error = (target - reconstructed).square().sum(dim=-1) / target.square().sum(
        dim=-1
    ).clamp_min(1e-12)
    cosine = F.cosine_similarity(target, reconstructed, dim=-1)
    return error, cosine


def _summary(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().float().cpu().numpy()
    return {
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _aggregate_layer(layer: int, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = sorted(set.intersection(*(set(item["methods"]) for item in datasets)))
    for method in methods:
        keys = set.intersection(*(set(item["methods"][method]["curve"]) for item in datasets))
        ordered_keys = sorted(
            keys,
            key=lambda value: (value == "native", int(value) if value.isdigit() else 0),
        )
        for key in ordered_keys:
            row: dict[str, Any] = {"layer": layer, "method": method, "k": key}
            for metric in ("nmse", "cosine"):
                for statistic in ("mean", "p10", "median", "p90"):
                    row[f"{metric}_{statistic}"] = float(np.mean([
                        item["methods"][method]["curve"][key][metric][statistic]
                        for item in datasets
                    ]))
            if key == "native":
                row["effective_k"] = float(np.mean([
                    item["methods"][method]["native_mean_active"] for item in datasets
                ]))
            elif method.startswith("sae"):
                row["effective_k"] = float(np.mean([
                    item["methods"][method]["effective_k"][key] for item in datasets
                ]))
            else:
                row["effective_k"] = int(key)
            rows.append(row)
    return rows


def _finish(
    output: Path,
    run_path: Path,
    run: dict[str, Any],
    resolved: dict[str, Any],
    layers: list[int],
) -> None:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        path = _layer_path(output, layer)
        if path.is_file():
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    _write_json(
        output / "results.json",
        {
            "experiment": resolved,
            "icalens_source": run.get("icalens_source"),
            "rows": rows,
        },
    )
    run["status"] = "complete"
    _write_json(run_path, run)


def _load_run(path: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        run = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if run.get("resolved") != resolved:
            # A failed run with no completed layers contains no reusable
            # measurements. Let it adopt corrected preset metadata so a
            # dependency or dataset-loader fix does not require manual cleanup.
            completed = any(
                item.get("status") == "complete"
                for item in run.get("layer_runs", {}).values()
                if isinstance(item, dict)
            )
            if completed:
                raise ValueError(
                    f"{path} belongs to a different reconstruction configuration"
                )
            return {"status": "running", "resolved": resolved, "layer_runs": {}}
        run["status"] = "running"
        return run
    return {"status": "running", "resolved": resolved, "layer_runs": {}}


def _layer_path(output: Path, layer: int) -> Path:
    return output / "layers" / f"layer_{layer:02d}.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_layers(value: str, available: Sequence[int]) -> list[int]:
    if value.strip().lower() == "all":
        return list(available)
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    missing = sorted(set(layers).difference(available))
    if not layers or missing:
        raise ValueError(f"invalid layers {missing}; available layers: {tuple(available)}")
    return list(dict.fromkeys(layers))


def _parse_capture_layers_at_once(value: str | int, *, selected_layer_count: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return selected_layer_count
    try:
        count = int(text)
    except ValueError as error:
        raise ValueError(
            "--capture-layers-at-once must be a positive integer or 'all'"
        ) from error
    if count < 1:
        raise ValueError("--capture-layers-at-once must be a positive integer or 'all'")
    return count


def _load_preset(name: str) -> dict[str, Any]:
    path = files("icalens.experiments").joinpath("configs", f"reconstruction_{name}.json")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
