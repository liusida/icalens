#!/usr/bin/env python3
"""Compare token-position effects in two ICA lenses on shared held-out text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
from matplotlib.colors import LogNorm
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._capture import capture_resid_post

LENS_NAMES = ("released", "skip-2")
POSITION_LABELS = ("0", "1", "2", "3+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-lens", type=Path, required=True)
    parser.add_argument("--skip-lens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--minimum-tokens", type=int, default=32)
    parser.add_argument("--extreme-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.documents <= 0 or args.context_length < 4 or args.extreme_k <= 0:
        raise ValueError(
            "documents and extreme-k must be positive; context-length must be at least 4"
        )

    lenses = {
        "released": ICALens.from_pretrained(args.released_lens),
        "skip-2": ICALens.from_pretrained(args.skip_lens),
    }
    released = lenses["released"]
    layers = tuple(released.available_layers)
    if tuple(lenses["skip-2"].available_layers) != layers:
        raise ValueError("lenses do not contain the same layers")
    if released.model_id != lenses["skip-2"].model_id:
        raise ValueError("lenses target different models")
    if released.model_revision != lenses["skip-2"].model_revision:
        raise ValueError("lenses target different model revisions")

    tokenizer = AutoTokenizer.from_pretrained(
        released.model_id, revision=released.model_revision, use_fast=True
    )
    documents = load_documents(args, tokenizer)
    model = cast(
        torch.nn.Module,
        load_model_to_cuda(
            AutoModelForCausalLM,
            released.model_id,
            revision=released.model_revision,
            device="cuda",
            dtype=torch.bfloat16,
            touch="auto",
            low_cpu_mem_usage=True,
        ),
    )
    model.eval()

    shape = (len(LENS_NAMES), len(layers), len(POSITION_LABELS))
    absolute_sum = np.zeros(shape, dtype=np.float64)
    square_sum = np.zeros(shape, dtype=np.float64)
    score_count = np.zeros(shape, dtype=np.int64)
    max_energy_sum = np.zeros(shape, dtype=np.float64)
    token_count = np.zeros(len(POSITION_LABELS), dtype=np.int64)
    top_values: dict[tuple[str, int], torch.Tensor] = {}
    top_is_early: dict[tuple[str, int], torch.Tensor] = {}

    for document in tqdm(documents, desc="Held-out documents", unit="doc"):
        ids = torch.tensor(document, dtype=torch.long, device="cuda").unsqueeze(0)
        positions = torch.arange(ids.shape[1], device="cuda")
        captured = capture_resid_post(
            model,
            model_inputs={"input_ids": ids},
            layers=layers,
            positions=positions,
        )
        bins = torch.clamp(positions, max=3)
        early = positions < 2
        for position_bin in range(4):
            token_count[position_bin] += int((bins == position_bin).sum())

        for layer_index, layer in enumerate(layers):
            activations = captured[layer].float()
            for lens_index, lens_name in enumerate(LENS_NAMES):
                scores = cast(torch.Tensor, lenses[lens_name].transform(activations, layer=layer))
                scores = scores.float()
                squared = scores.square()
                maximum_energy = squared.max(dim=1).values / squared.sum(dim=1).clamp_min(1e-30)
                for position_bin in range(4):
                    selected = bins == position_bin
                    selected_scores = scores[selected]
                    absolute_sum[lens_index, layer_index, position_bin] += float(
                        selected_scores.abs().sum()
                    )
                    square_sum[lens_index, layer_index, position_bin] += float(
                        selected_scores.square().sum()
                    )
                    score_count[lens_index, layer_index, position_bin] += selected_scores.numel()
                    max_energy_sum[lens_index, layer_index, position_bin] += float(
                        maximum_energy[selected].sum()
                    )

                key = (lens_name, layer)
                values = scores.abs().T
                flags = early.unsqueeze(0).expand(values.shape[0], -1)
                if key in top_values:
                    values = torch.cat((top_values[key], values), dim=1)
                    flags = torch.cat((top_is_early[key], flags), dim=1)
                keep = min(args.extreme_k, values.shape[1])
                selected_values, selected_indices = torch.topk(values, k=keep, dim=1)
                top_values[key] = selected_values
                top_is_early[key] = flags.gather(1, selected_indices)

    rms = np.sqrt(square_sum / score_count)
    mean_absolute = absolute_sum / score_count
    mean_max_energy = max_energy_sum / token_count[None, None, :]
    total_tokens = int(token_count.sum())
    early_tokens = int(token_count[:2].sum())
    early_baseline = early_tokens / total_tokens

    args.output.mkdir(parents=True, exist_ok=True)
    position_rows = write_position_metrics(
        args.output, layers, rms, mean_absolute, mean_max_energy, token_count
    )
    extreme_rows = write_extreme_metrics(
        args.output, layers, top_is_early, args.extreme_k, early_baseline
    )
    plot_overview(args.output, layers, rms, mean_max_energy, extreme_rows, early_baseline)
    plot_layers(args.output, layers, rms, mean_max_energy, extreme_rows, early_baseline)

    summary = {
        "comparison": "first-position effects on shared unframed held-out documents",
        "dataset": {
            "repo_id": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
        },
        "documents": len(documents),
        "context_length": args.context_length,
        "tokens": total_tokens,
        "position_token_counts": dict(zip(POSITION_LABELS, token_count.tolist(), strict=True)),
        "extreme_k": args.extreme_k,
        "early_position_baseline": early_baseline,
        "layers": list(layers),
        "lenses": {name: str(path.resolve()) for name, path in zip(
            LENS_NAMES, (args.released_lens, args.skip_lens), strict=True
        )},
        "position_metrics": position_rows,
        "extreme_metrics": extreme_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def load_documents(args: argparse.Namespace, tokenizer: Any) -> list[list[int]]:
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    documents: list[list[int]] = []
    for record in dataset:
        text = str(record[args.text_field])
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) < args.minimum_tokens:
            continue
        documents.append([int(value) for value in ids[: args.context_length]])
        if len(documents) == args.documents:
            break
    if len(documents) < args.documents:
        raise ValueError(f"found only {len(documents)} documents with enough tokens")
    return documents


def write_position_metrics(
    output: Path,
    layers: tuple[int, ...],
    rms: np.ndarray,
    mean_absolute: np.ndarray,
    mean_max_energy: np.ndarray,
    token_count: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for lens_index, lens_name in enumerate(LENS_NAMES):
        for layer_index, layer in enumerate(layers):
            for position_bin, position_label in enumerate(POSITION_LABELS):
                rows.append(
                    {
                        "lens": lens_name,
                        "layer": layer,
                        "position": position_label,
                        "tokens": int(token_count[position_bin]),
                        "score_rms": float(rms[lens_index, layer_index, position_bin]),
                        "mean_absolute_score": float(
                            mean_absolute[lens_index, layer_index, position_bin]
                        ),
                        "mean_max_component_energy": float(
                            mean_max_energy[lens_index, layer_index, position_bin]
                        ),
                    }
                )
    write_csv(output / "position-metrics.csv", rows)
    return rows


def write_extreme_metrics(
    output: Path,
    layers: tuple[int, ...],
    top_is_early: dict[tuple[str, int], torch.Tensor],
    extreme_k: int,
    early_baseline: float,
) -> list[dict[str, Any]]:
    rows = []
    for lens_name in LENS_NAMES:
        for layer in layers:
            counts = top_is_early[(lens_name, layer)].sum(dim=1).cpu().numpy()
            for component, count in enumerate(counts):
                fraction = float(count / extreme_k)
                rows.append(
                    {
                        "lens": lens_name,
                        "layer": layer,
                        "component": component,
                        "early_extreme_count": int(count),
                        "extreme_k": extreme_k,
                        "early_extreme_fraction": fraction,
                        "early_enrichment": fraction / early_baseline,
                    }
                )
    write_csv(output / "component-extreme-position-enrichment.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    output: Path,
    layers: tuple[int, ...],
    rms: np.ndarray,
    mean_max_energy: np.ndarray,
    extreme_rows: list[dict[str, Any]],
    early_baseline: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for lens_index, lens_name in enumerate(LENS_NAMES):
        score_ratio = rms[lens_index, :, :3] / rms[lens_index, :, 3, None]
        energy_ratio = (
            mean_max_energy[lens_index, :, :3] / mean_max_energy[lens_index, :, 3, None]
        )
        plot_heatmap(
            axes[0, lens_index],
            score_ratio,
            layers,
            f"{lens_name}: score RMS / pos 3+",
            value_range=(0.5, 200),
        )
        plot_heatmap(
            axes[1, lens_index],
            energy_ratio,
            layers,
            f"{lens_name}: max-component energy / pos 3+",
            value_range=(0.05, 2),
        )
    fig.suptitle("First-token effects across GPT-2 layers (unframed held-out text)")
    fig.savefig(output / "position-effects-all-layers.png", dpi=180)
    fig.savefig(output / "position-effects-all-layers.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for lens_name in LENS_NAMES:
        means = []
        saturated = []
        for layer in layers:
            values = [
                row["early_extreme_fraction"]
                for row in extreme_rows
                if row["lens"] == lens_name and row["layer"] == layer
            ]
            means.append(float(np.mean(values)))
            saturated.append(float(np.mean(np.asarray(values) == 1)))
        axes[0].plot(layers, means, "o-", label=lens_name)
        axes[1].plot(layers, saturated, "o-", label=lens_name)
    axes[0].axhline(
        early_baseline, color="black", linestyle=":", label="token baseline"
    )
    axes[0].set(
        title="Mean position-0/1 fraction in top-|score| sets",
        xlabel="Layer",
        ylabel="Fraction of top 100",
        xticks=layers,
    )
    axes[1].set(
        title="Components whose entire top 100 are positions 0–1",
        xlabel="Layer",
        ylabel="Fraction of components",
        xticks=layers,
    )
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "extreme-position-enrichment-all-layers.png", dpi=180)
    fig.savefig(output / "extreme-position-enrichment-all-layers.pdf")
    plt.close(fig)


def plot_heatmap(
    ax: Any,
    values: np.ndarray,
    layers: tuple[int, ...],
    title: str,
    *,
    value_range: tuple[float, float],
) -> None:
    image = ax.imshow(values, aspect="auto", norm=LogNorm(*value_range))
    ax.set(title=title, xlabel="Position", ylabel="Layer")
    ax.set_xticks(range(3), POSITION_LABELS[:3])
    ax.set_yticks(range(len(layers)), layers)
    plt.colorbar(image, ax=ax, shrink=0.8)


def plot_layers(
    output: Path,
    layers: tuple[int, ...],
    rms: np.ndarray,
    mean_max_energy: np.ndarray,
    extreme_rows: list[dict[str, Any]],
    early_baseline: float,
) -> None:
    layer_root = output / "layers"
    layer_root.mkdir(exist_ok=True)
    for layer_index, layer in enumerate(layers):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for lens_index, lens_name in enumerate(LENS_NAMES):
            axes[0].plot(POSITION_LABELS, rms[lens_index, layer_index], "o-", label=lens_name)
            axes[1].plot(
                POSITION_LABELS,
                mean_max_energy[lens_index, layer_index],
                "o-",
                label=lens_name,
            )
            values = [
                row["early_extreme_fraction"]
                for row in extreme_rows
                if row["lens"] == lens_name and row["layer"] == layer
            ]
            axes[2].hist(
                values,
                bins=np.linspace(0, max(0.2, max(values)), 31),
                alpha=0.5,
                label=lens_name,
            )
        axes[0].set(title="Score RMS", xlabel="Token position", ylabel="RMS")
        axes[1].set(
            title="Maximum component energy share",
            xlabel="Token position",
            ylabel="Mean share",
        )
        axes[2].axvline(early_baseline, color="black", linestyle=":", label="token baseline")
        axes[2].set(
            title="Positions 0–1 in component top-|score| set",
            xlabel="Fraction of extremes",
            ylabel="Components",
        )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(f"GPT-2 layer {layer}: released vs no-framing skip-2")
        fig.savefig(layer_root / f"layer-{layer:02d}.png", dpi=180)
        fig.savefig(layer_root / f"layer-{layer:02d}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
