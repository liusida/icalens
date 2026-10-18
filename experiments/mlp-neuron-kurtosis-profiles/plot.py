#!/usr/bin/env python3
"""Plot the all-neuron kurtosis and skewness distributions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/main"
DEFAULT_OUTPUT = HERE / "figures/neuron-statistics.png"
EXPECTED_FORMAT = "icalens.mlp_neuron_kurtosis_profiles"
EXPECTED_SCHEMA_VERSION = 1
COLOR = "#3D5F99"
REFERENCE_COLOR = "#777777"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing figure.")
    return parser.parse_args()


def load_statistics(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    run_path = root / "run.json"
    statistics_path = root / "neuron-statistics.npz"
    if not run_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError(f"run does not yet contain neuron statistics: {root}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    resolved = run.get("resolved", {})
    if (
        resolved.get("format") != EXPECTED_FORMAT
        or resolved.get("schema_version") != EXPECTED_SCHEMA_VERSION
    ):
        raise ValueError(f"not a compatible MLP-neuron run: {root}")
    with np.load(statistics_path, allow_pickle=False) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    count = int(values.get("count", -1))
    neurons = values.get("neuron")
    required = ("mean", "variance", "skewness", "excess_kurtosis")
    if (
        count != int(resolved["statistics_tokens"])
        or neurons is None
        or neurons.shape != (12288,)
        or not np.array_equal(neurons, np.arange(12288))
        or any(key not in values or values[key].shape != neurons.shape for key in required)
        or any(not np.isfinite(values[key]).all() for key in required)
    ):
        raise ValueError(f"invalid or incomplete neuron statistics: {statistics_path}")
    if np.any(values["excess_kurtosis"] <= 0):
        raise ValueError("log-scaled kurtosis histogram requires positive values")
    return run, values


def paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def render(run: dict, values: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    kurtosis = values["excess_kurtosis"]
    skewness = values["skewness"]
    layer = int(run["resolved"]["layer"])
    kurtosis_bins = np.geomspace(kurtosis.min(), kurtosis.max() * 1.001, 55)
    skewness_bins = np.linspace(skewness.min(), skewness.max() * 1.001, 55)
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
        for axis in axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(axis="y", color="#E3E3E3", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.set_ylabel("Neurons")

        axes[0].hist(
            kurtosis,
            bins=kurtosis_bins,
            color=COLOR,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.25,
        )
        axes[0].set_xscale("log")
        axes[0].axvline(
            np.median(kurtosis),
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"median = {np.median(kurtosis):.2f}",
        )
        axes[0].set_title("A · Excess kurtosis", loc="left", fontweight="bold")
        axes[0].set_xlabel("Excess kurtosis")
        axes[0].legend(frameon=False, fontsize=7.0)

        axes[1].hist(
            skewness,
            bins=skewness_bins,
            color=COLOR,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.25,
        )
        axes[1].axvline(0.0, color="#B8B8B8", linewidth=0.7)
        axes[1].axvline(
            np.median(skewness),
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"median = {np.median(skewness):.3f}",
        )
        axes[1].set_title("B · Skewness", loc="left", fontweight="bold")
        axes[1].set_xlabel("Skewness")
        axes[1].legend(frameon=False, fontsize=7.0)

        figure.suptitle(
            f"Qwen 3.5 9B Base · layer {layer} · {len(kurtosis):,} MLP neurons",
            fontsize=8.5,
            fontweight="bold",
            y=0.995,
        )
        figure.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.82, wspace=0.28)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"figure already exists; pass --force to replace it: {output}")
    run, values = load_statistics(root)
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        render(run, values, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
