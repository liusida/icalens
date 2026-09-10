#!/usr/bin/env python3
"""Plot complete and partial official TPP results across models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    ("ica_custom_sae", "ICA", "#3D5F99", "^"),
    ("sae_custom_sae", "SAE", "#B45F4D", "o"),
    ("unfitted_ica_custom_sae", "Unfitted ICA", "#8FA9CC", "v"),
    ("untrained_sae_matched_l0_custom_sae", "Untrained SAE", "#D29A8E", "s"),
)
DRAW_ORDER = tuple(reversed(METHODS))
MODELS = (
    ("gpt2", "GPT-2 Small", (6, 10)),
    ("gemma2", "Gemma 2 2B", (12, 20)),
    ("qwen35-9b", "Qwen 3.5 9B Base", (12, 20)),
)
METRICS = (
    ("total_metric", "TPP score"),
    ("intended_diff_only", "Intended change"),
    ("unintended_diff_only", "Unintended change"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/tpp/official/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tpp/official/figures/tpp-model-comparison-partial.png"),
    )
    return parser.parse_args()


def load_available(root: Path, layers: tuple[int, ...]) -> dict[str, dict[int, dict]]:
    available: dict[str, dict[int, dict]] = {}
    for method_id, *_ in METHODS:
        by_layer = {}
        for layer in layers:
            path = (
                root
                / "layers"
                / f"layer_{layer:02d}"
                / "saebench"
                / "tpp"
                / f"{method_id}_eval_results.json"
            )
            if path.is_file():
                by_layer[layer] = json.loads(path.read_text(encoding="utf-8"))
        if by_layer:
            available[method_id] = by_layer
    return available


def metric_value(payload: dict, budget: int, suffix: str) -> float:
    key = f"tpp_threshold_{budget}_{suffix}"
    summary = payload["eval_result_metrics"]["tpp_metrics"].get(key)
    raw = payload["eval_result_unstructured"]
    dataset_means = [
        statistics.mean(float(class_result[key]) for class_result in classes.values())
        for classes in raw.values()
    ]
    reconstructed = statistics.mean(dataset_means)
    if summary is not None and not np.isclose(float(summary), reconstructed, atol=1e-7):
        raise ValueError(f"stored and reconstructed values disagree for {key}")
    return reconstructed


def plot_method(
    axis: plt.Axes,
    payloads: dict[int, dict],
    *,
    suffix: str,
    label: str,
    color: str,
    marker: str,
    zorder: int,
) -> None:
    first = next(iter(payloads.values()))
    budgets = [int(value) for value in first["eval_config"]["n_values"]]
    layer_values = np.asarray(
        [
            [metric_value(payload, budget, suffix) for budget in budgets]
            for payload in payloads.values()
        ]
    )
    mean = layer_values.mean(axis=0)
    axis.plot(
        budgets,
        mean,
        color=color,
        marker=marker,
        linewidth=1.35,
        markersize=3.4,
        label=label,
        zorder=zorder,
    )


def main() -> None:
    args = parse_args()
    model_data = [
        (model_name, title, layers, load_available(args.results / model_name, layers))
        for model_name, title, layers in MODELS
    ]
    if not all(data for _, _, _, data in model_data):
        raise ValueError("at least one model has no available TPP method results")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    figure = plt.figure(figsize=(11.2, 4.8))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 0.72), hspace=0.48, wspace=0.42)
    axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for index in range(3):
        axes.append(
            (
                figure.add_subplot(grid[0, 2 * index : 2 * index + 2]),
                figure.add_subplot(grid[1, 2 * index]),
                figure.add_subplot(grid[1, 2 * index + 1]),
            )
        )

    for model_index, (_, title, expected_layers, data) in enumerate(model_data):
        complete_layers = sorted(set.intersection(*(set(v) for v in data.values())))
        complete_methods = sum(set(v) == set(expected_layers) for v in data.values())
        subtitle = (
            f"mean of L{expected_layers[0]} and L{expected_layers[1]}"
            if complete_methods == len(METHODS)
            else f"partial: L{','.join(map(str, complete_layers)) or '—'}; "
            f"{len(data)}/{len(METHODS)} methods"
        )
        top, intended, unintended = axes[model_index]
        top.set_title(f"{title}\n{subtitle}", fontweight="bold", loc="left")
        intended.set_title("Intended", fontweight="bold", loc="left")
        unintended.set_title("Unintended", fontweight="bold", loc="left")
        for metric_index, axis in enumerate((top, intended, unintended)):
            suffix, _ = METRICS[metric_index]
            for draw_index, (method_id, label, color, marker) in enumerate(DRAW_ORDER):
                if method_id in data:
                    plot_method(
                        axis,
                        data[method_id],
                        suffix=suffix,
                        label=label,
                        color=color,
                        marker=marker,
                        zorder=2 + draw_index,
                    )
            axis.axhline(0, color="0.65", linewidth=0.7, zorder=0)
            axis.set_xscale("log")
            axis.set_xticks((1, 2, 5, 10, 20, 50, 100))
            axis.set_xticklabels(("1", "2", "5", "10", "20", "50", "100"), rotation=38)
            axis.grid(axis="y", color="0.88", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        top.set_ylabel("TPP score" if model_index == 0 else "")
        intended.set_ylabel("TPP effect" if model_index == 0 else "")

    all_axes = [axis for group in axes for axis in group]
    y_min = min(axis.get_ylim()[0] for axis in all_axes)
    y_max = max(axis.get_ylim()[1] for axis in all_axes)
    for axis in all_axes:
        axis.set_ylim(y_min, y_max)
    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    ordered_labels = [label for _, label, _, _ in METHODS if label in by_label]
    figure.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    figure.supxlabel("Top-k perturbed features", y=0.025)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.16, top=0.82)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
