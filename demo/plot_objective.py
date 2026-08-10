"""Plot saved FastICA objective-percentile curves from an ICA Lens artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

GAUSSIAN_LOGCOSH_BASELINE = 0.375


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lens", type=Path, help="Local ICA Lens artifact directory.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices; defaults to the first available layers.",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=4,
        help="Number of first available layers to plot when --layers is omitted (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <lens>/objective-curves.png).",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true", help="Also open the Matplotlib window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lens_dir = args.lens.expanduser().resolve()
    manifest_path = lens_dir / "icalens.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = sorted(int(layer) for layer in manifest["layers"])
    if args.layers is None:
        if args.first <= 0:
            raise ValueError("--first must be positive")
        layers = available[: args.first]
    else:
        layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
    if not layers:
        raise ValueError("no layers selected")
    missing = [layer for layer in layers if layer not in available]
    if missing:
        raise ValueError(f"layers not available: {missing}; available layers: {available}")

    columns = min(2, len(layers))
    rows = math.ceil(len(layers) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 4.6 * rows),
        squeeze=False,
        sharex=True,
    )
    for axis, layer in zip(axes.flat, layers, strict=False):
        plot_layer(axis, layer, manifest["layers"][str(layer)]["fitting"])
    for axis in axes.flat[len(layers) :]:
        axis.remove()
    figure.suptitle("FastICA objective distribution across components")
    figure.supxlabel("Iteration")
    figure.supylabel("Mean contrast over fitting tokens")
    figure.tight_layout()

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else lens_dir / "objective-curves.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved objective plot to {output}")
    if args.show:
        plt.show()
    plt.close(figure)


def plot_layer(axis: Any, layer: int, fitting: dict[str, Any]) -> None:
    history = fitting.get("objective_history")
    if not isinstance(history, dict):
        raise ValueError(f"layer {layer} has no objective history")
    iterations = np.asarray(history["iterations"], dtype=np.int64)
    percentiles = np.asarray(history["percentiles"], dtype=np.int64)
    values = np.asarray(history["values"], dtype=np.float64)
    if values.shape != (len(iterations), len(percentiles)):
        raise ValueError(f"layer {layer} has inconsistent objective-history dimensions")
    collapsed = bool(np.all(values == values[:, :1]))
    if collapsed:
        print(
            f"Warning: layer {layer} has identical percentile columns; "
            "this artifact contains legacy scalar objective history."
        )

    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, 5))
    for index, lower in enumerate((0, 10, 20, 30, 40)):
        upper = 100 - lower
        low_index = percentile_index(percentiles, lower, layer)
        high_index = percentile_index(percentiles, upper, layer)
        axis.fill_between(
            iterations,
            values[:, low_index],
            values[:, high_index],
            color=colors[index],
            alpha=0.22,
            linewidth=0,
            label=f"p{lower}–p{upper}",
        )
    median_index = percentile_index(percentiles, 50, layer)
    axis.plot(iterations, values[:, median_index], color="#172554", linewidth=1.8, label="p50")
    suffix = " (scalar history)" if collapsed else ""
    axis.set_title(f"Layer {layer}{suffix}")
    axis.set_ylim(0, GAUSSIAN_LOGCOSH_BASELINE)
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8, frameon=False)


def percentile_index(percentiles: np.ndarray[Any, Any], value: int, layer: int) -> int:
    matches = np.flatnonzero(percentiles == value)
    if len(matches) != 1:
        raise ValueError(f"layer {layer} objective history does not contain p{value}")
    return int(matches[0])


if __name__ == "__main__":
    main()
