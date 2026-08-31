"""Plot layerwise distributions of component-level gradient ERF."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BIN_EDGES = np.asarray([1.0, 2.0, 3.0, 6.0, 11.0, np.inf])
BIN_LABELS = ("1–<2", "2–<3", "3–<6", "6–<11", "≥11")
BIN_COLORS = ("#4F8A63", "#8FBE85", "#E7B84B", "#CC7445", "#76558D")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure erf-gradient", description=__doc__
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel-titles", default=None, help="Comma-separated titles.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.expanduser().resolve()
    output = (
        experiment / "figures" / "erf-gradient"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    paths = render(
        experiment,
        output_prefix=output,
        panel_titles=args.panel_titles,
        force=args.force,
    )
    for path in paths:
        print(path)


def render(
    experiment: Path,
    *,
    output_prefix: Path,
    panel_titles: str | None,
    force: bool,
) -> list[Path]:
    run_path = experiment / "run.json"
    summary_path = experiment / "summary.csv"
    if not run_path.is_file() or not summary_path.is_file():
        raise ValueError(f"incomplete gradient ERF experiment: {experiment}")
    import json

    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"gradient ERF experiment is not complete: {experiment}")
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    labels = list(run["resolved"].get("lens_order", run["resolved"]["lenses"]))
    titles = _titles(panel_titles, labels, run)
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
    for path in outputs:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(1, len(labels), figsize=(6.9, 2.75), sharey=True, squeeze=False)
        for panel_index, (axis, label, title) in enumerate(
            zip(axes.ravel(), labels, titles, strict=True)
        ):
            model_rows = [row for row in rows if row["model"] == label]
            layers = sorted({int(row["layer"]) for row in model_rows})
            fractions = np.zeros((len(BIN_LABELS), len(layers)), dtype=float)
            for column, layer in enumerate(layers):
                values = np.asarray(
                    [
                        float(row["gradient_erf_median"])
                        for row in model_rows
                        if int(row["layer"]) == layer
                    ]
                )
                if len(values) != int(run["resolved"]["components_per_layer"]):
                    raise ValueError(f"{label} layer {layer} has incomplete component results")
                bins = np.searchsorted(BIN_EDGES, values, side="right") - 1
                fractions[:, column] = [np.mean(bins == index) for index in range(len(BIN_LABELS))]
            bottom = np.zeros(len(layers))
            for values, color in zip(fractions, BIN_COLORS, strict=True):
                axis.bar(
                    layers,
                    values,
                    bottom=bottom,
                    width=0.86,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                )
                bottom += values
            axis.set_title(title, loc="left", fontweight="bold", pad=3)
            axis.set_ylim(0, 1)
            axis.set_xlim(min(layers) - 0.6, max(layers) + 0.6)
            tick_step = max(1, round(len(layers) / 6))
            axis.set_xticks(layers[::tick_step])
            axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if panel_index == 0:
                axis.set_ylabel("Fraction of components")
        figure.supxlabel("Layer", y=0.06)
        figure.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for color, label in zip(BIN_COLORS, BIN_LABELS, strict=True)
            ],
            loc="upper center",
            ncol=len(BIN_LABELS),
            frameon=False,
            bbox_to_anchor=(0.5, 1.015),
        )
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.82, wspace=0.18)
        figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
        figure.savefig(outputs[1], bbox_inches="tight")
        plt.close(figure)
    outputs[2].write_text(_caption(labels, titles, run), encoding="utf-8")
    return outputs


def _titles(panel_titles: str | None, labels: list[str], run: dict[str, Any]) -> list[str]:
    if panel_titles:
        values = [value.strip() for value in panel_titles.split(",")]
        if len(values) != len(labels):
            raise ValueError("--panel-titles must provide one title per model")
        return values
    known = {
        "openai-community/gpt2": "GPT-2 small",
        "google/gemma-2-2b": "Gemma 2 2B",
        "Qwen/Qwen3.5-9B-Base": "Qwen 3.5 9B Base",
    }
    return [
        known.get(run["resolved"]["lenses"][label]["model"]["repo_id"], label) for label in labels
    ]


def _caption(labels: list[str], titles: list[str], run: dict[str, Any]) -> str:
    count = int(run["resolved"]["components_per_layer"])
    seed = int(run["resolved"]["seed"])
    panels = ", ".join(f"{label} ({title})" for label, title in zip(labels, titles, strict=True))
    return (
        "Layerwise distribution of gradient effective receptive field (ERF_grad). "
        f"Each layer contains {count} randomly sampled components (seed {seed}); each "
        "component is summarized by the median influence-weighted geometric token distance "
        "over its stored dominant-tail occurrences. Bars show fractions in the intervals "
        "[1,2), [2,3), [3,6), [6,11), and [11,infinity). "
        f"Panels: {panels}.\n"
    )


def _paper_style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
