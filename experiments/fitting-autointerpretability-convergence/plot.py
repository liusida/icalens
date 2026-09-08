#!/usr/bin/env python3
"""Plot individual and aggregate autointerpretability convergence trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from prepare import EVALUATED_CHECKPOINTS, LAYERS, N_FEATURES

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs/evaluated-tinker"
DEFAULT_PREPARATION = ROOT / "runs/prepared"
DEFAULT_OUTPUT = ROOT / "figures"
SCORE_FILE = re.compile(r"feature_(\d+)\.json")
ICA_BLUE = "#3D5F99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_features(preparation: Path) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in LAYERS:
            path = (
                preparation
                / f"iter-{iteration:03d}"
                / f"layer_{layer:02d}"
                / "ica"
                / "selection.json"
            )
            selection = json.loads(path.read_text(encoding="utf-8"))
            accepted = selection.get("accepted")
            if not isinstance(accepted, list) or len(accepted) != N_FEATURES:
                raise ValueError(f"expected {N_FEATURES} accepted features: {path}")
            mapping = {int(row["feature"]): position for position, row in enumerate(accepted)}
            if len(mapping) != N_FEATURES:
                raise ValueError(f"feature IDs are not unique: {path}")
            key = layer
            if key in result and result[key] != mapping:
                raise ValueError(f"cohort order changed at iteration {iteration}, layer {layer}")
            result[key] = mapping
    return result


def read_rows(input_root: Path, preparation: Path) -> list[dict[str, Any]]:
    feature_positions = selected_features(preparation)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for iteration in EVALUATED_CHECKPOINTS:
        for layer in LAYERS:
            directory = input_root / f"iter-{iteration:03d}" / f"layer_{layer:02d}/ica/results"
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("feature_*.json")):
                match = SCORE_FILE.fullmatch(path.name)
                if match is None:
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") != "complete":
                    continue
                feature = int(match.group(1))
                if feature not in feature_positions[layer]:
                    raise ValueError(f"unexpected feature C{feature}: {path}")
                position = feature_positions[layer][feature]
                key = (iteration, layer, position)
                if key in seen:
                    raise ValueError(f"duplicate result for {key}: {path}")
                scores = {
                    name: float(record[name])
                    for name in ("combined_score", "top_score", "random_score")
                }
                if not all(math.isfinite(value) for value in scores.values()):
                    raise ValueError(f"non-finite score: {path}")
                seen.add(key)
                rows.append(
                    {
                        "iteration": iteration,
                        "layer": layer,
                        "cohort_position": position,
                        "feature": feature,
                        **scores,
                    }
                )
    if not rows:
        raise ValueError(f"no completed feature results found under {input_root}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "iteration", "layer", "cohort_position", "feature",
        "combined_score", "top_score", "random_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (
            row["layer"], row["cohort_position"], row["iteration"]
        )))


def rc_params() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 7,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="0.89", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_xscale("symlog", linthresh=2, linscale=1)
    axis.set_xlim(-0.15, 215)
    axis.set_xticks(EVALUATED_CHECKPOINTS)
    axis.set_xticklabels([str(value) for value in EVALUATED_CHECKPOINTS], rotation=45)
    axis.set_ylim(-0.15, 1.0)


def trajectory_matrix(rows: list[dict[str, Any]], layer: int) -> np.ndarray:
    matrix = np.full((N_FEATURES, len(EVALUATED_CHECKPOINTS)), np.nan)
    iteration_index = {value: index for index, value in enumerate(EVALUATED_CHECKPOINTS)}
    for row in rows:
        if row["layer"] == layer:
            matrix[row["cohort_position"], iteration_index[row["iteration"]]] = row[
                "combined_score"
            ]
    return matrix


def save_figure(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_detailed(plt: Any, rows: list[dict[str, Any]], stem: Path) -> None:
    with plt.rc_context(rc_params()):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), sharex=True, sharey=True)
        for panel, (axis, layer) in enumerate(zip(axes, LAYERS, strict=True)):
            matrix = trajectory_matrix(rows, layer)
            for values in matrix:
                axis.plot(
                    EVALUATED_CHECKPOINTS,
                    values,
                    color=ICA_BLUE,
                    linewidth=0.55,
                    alpha=0.28,
                )
            style_axis(axis)
            axis.set_title(f"{'AB'[panel]} · Qwen 3.5 9B Base · L{layer}", loc="left")
            axis.set_xlabel("FastICA iteration")
        axes[0].set_ylabel("Autointerpretability score")
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.91, wspace=0.10)
        save_figure(figure, stem)
        plt.close(figure)


def plot_aggregate(plt: Any, rows: list[dict[str, Any]], stem: Path) -> dict[int, list[int]]:
    counts: dict[int, list[int]] = {}
    with plt.rc_context(rc_params()):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), sharex=True, sharey=True)
        for panel, (axis, layer) in enumerate(zip(axes, LAYERS, strict=True)):
            matrix = trajectory_matrix(rows, layer)
            n = np.sum(np.isfinite(matrix), axis=0)
            counts[layer] = n.tolist()
            mean = np.full(matrix.shape[1], np.nan)
            sem = np.full(matrix.shape[1], np.nan)
            for index in range(matrix.shape[1]):
                values = matrix[:, index]
                values = values[np.isfinite(values)]
                if len(values):
                    mean[index] = values.mean()
                if len(values) > 1:
                    sem[index] = values.std(ddof=1) / np.sqrt(len(values))
            color = ICA_BLUE
            axis.fill_between(
                EVALUATED_CHECKPOINTS,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                EVALUATED_CHECKPOINTS,
                mean,
                color=color,
                linewidth=1.5,
                marker="D",
                markersize=2.8,
            )
            partial = (n > 0) & (n < N_FEATURES)
            axis.scatter(
                np.asarray(EVALUATED_CHECKPOINTS)[partial],
                mean[partial],
                s=18,
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                zorder=4,
            )
            style_axis(axis)
            axis.set_title(f"{'AB'[panel]} · Qwen 3.5 9B Base · L{layer}", loc="left")
            axis.set_xlabel("FastICA iteration")
        axes[0].set_ylabel("Mean autointerpretability score")
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.91, wspace=0.10)
        save_figure(figure, stem)
        plt.close(figure)
    return counts


def main() -> None:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    preparation = args.preparation.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detailed = output / "autointerpretability-trajectories"
    aggregate = output / "autointerpretability-convergence"
    targets = [
        detailed.with_suffix(suffix)
        for suffix in (".png", ".pdf", ".txt")
    ] + [
        aggregate.with_suffix(suffix)
        for suffix in (".png", ".pdf", ".txt")
    ] + [output / "autointerpretability-convergence-data.csv"]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to replace existing outputs without --force: "
            + ", ".join(map(str, existing))
        )
    rows = read_rows(input_root, preparation)
    write_csv(output / "autointerpretability-convergence-data.csv", rows)
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib.pyplot as plt

        plot_detailed(plt, rows, detailed)
        counts = plot_aggregate(plt, rows, aggregate)
    complete_positions = sum(
        all(
            any(
                row["cohort_position"] == position
                and row["iteration"] == iteration
                and row["layer"] == layer
                for row in rows
            )
            for iteration in EVALUATED_CHECKPOINTS
            for layer in LAYERS
        )
        for position in range(N_FEATURES)
    )
    detailed.with_suffix(".txt").write_text(
        "Individual combined autointerpretability scores for the same 50 persistent "
        "FastICA rows at Qwen layers 15 and 31. Thin lines identify cohort positions; "
        "the iteration axis uses symmetric-log spacing to expose the densely sampled "
        "early trajectory, and missing evaluations remain gaps. Current complete "
        "matched trajectories: "
        f"{complete_positions}/{N_FEATURES}.\n",
        encoding="utf-8",
    )
    count_text = "; ".join(
        f"L{layer}: " + ", ".join(
            f"{iteration}={count}"
            for iteration, count in zip(EVALUATED_CHECKPOINTS, counts[layer], strict=True)
        )
        for layer in LAYERS
    )
    aggregate.with_suffix(".txt").write_text(
        "Mean combined autointerpretability score across available matched FastICA rows. "
        "Bands show mean ± one standard error when at least two rows are available; "
        "hollow points have fewer than the nominal 50 rows. The iteration axis uses "
        "symmetric-log spacing. Counts by iteration: "
        f"{count_text}.\n",
        encoding="utf-8",
    )
    print(f"Wrote detailed and aggregate figures to {output}")
    print(f"Complete matched trajectories: {complete_positions}/{N_FEATURES}")


if __name__ == "__main__":
    main()
