#!/usr/bin/env python3
"""Aggregate per-layer ICA direction-comparison summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in args.results_root.glob("gpt2-layer-*/summary.json"):
        with path.open() as handle:
            summaries.append(json.load(handle))
    summaries.sort(key=lambda item: item["layer"])
    if not summaries:
        raise FileNotFoundError(f"No layer summaries found under {args.results_root}")

    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "component_count",
        "mean",
        "median",
        "p05",
        "minimum",
        "fraction_ge_0.70",
        "fraction_ge_0.90",
        "fraction_ge_0.95",
    ]
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item[field] for field in fields} for item in summaries)

    with (args.output / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")

    layers = [item["layer"] for item in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(layers, [item["mean"] for item in summaries], "o-", label="Mean")
    axes[0].plot(layers, [item["median"] for item in summaries], "o-", label="Median")
    axes[0].plot(layers, [item["p05"] for item in summaries], "o-", label="5th percentile")
    axes[0].set(title="Matched absolute cosine similarity", xlabel="Layer", ylabel="Similarity")
    axes[0].set_xticks(layers)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for threshold in ("0.70", "0.90", "0.95"):
        axes[1].plot(
            layers,
            [item[f"fraction_ge_{threshold}"] for item in summaries],
            "o-",
            label=f"≥ {threshold}",
        )
    axes[1].set(title="Fraction of matched components", xlabel="Layer", ylabel="Fraction")
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("GPT-2 released lens vs no-framing, skip-2 lens")
    fig.savefig(args.output / "all-layers.png", dpi=180)
    fig.savefig(args.output / "all-layers.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
