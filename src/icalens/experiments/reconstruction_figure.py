"""Create paper-ready held-out reconstruction figures."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import tempfile
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icalens experiment figure reconstruction", description=__doc__
    )
    parser.add_argument("experiment", nargs="+", help="Reconstruction result directories.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--format",
        default="png",
        help="Comma-separated output formats: png or pdf (default: png).",
    )
    parser.add_argument("--panel-titles", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiments = [Path(item).expanduser().resolve() for item in args.experiment]
    payloads = [_load(path) for path in experiments]
    formats = [item.strip() for item in args.format.split(",") if item.strip()]
    if not formats or any(item not in {"png", "pdf"} for item in formats):
        raise ValueError("--format must contain png, pdf, or both")
    output = experiments[0] / "figures" if args.output is None else args.output.resolve()
    titles = _titles(args.panel_titles, payloads)
    outputs = render(payloads, titles=titles, output=output, formats=formats, force=args.force)
    for path in outputs:
        print(path)


def render(
    payloads: Sequence[dict[str, Any]],
    *,
    titles: Sequence[str],
    output: Path,
    formats: Sequence[str],
    force: bool,
) -> list[Path]:
    cache = Path(tempfile.gettempdir()) / "icalens-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    style = {
        "ica": ("ICA top-k", "#3D5F99", "D"),
        "sae": ("SAE top-k", "#B45F4D", "o"),
        "sae_context_64": ("SAE top-k · positions <64", "#B45F4D", "o"),
        "pca": ("PCA top-k", "#5B8C6A", "^"),
        "random": ("Random top-k", "#777777", "s"),
    }
    outputs: list[Path] = []
    for metric, ylabel in (("nmse", "Normalized MSE"), ("cosine", "Cosine similarity")):
        overall = [
            (title, payload.get("rows", []))
            for payload, title in zip(payloads, titles, strict=True)
        ]
        outputs.extend(_render_grid(
            plt, overall, metric=metric, ylabel=ylabel, style=style, output=output,
            stem=f"reconstruction-{metric}", formats=formats, force=force,
            caption=("Held-out token-level top-k reconstruction, averaged equally across "
                     "selected layers and evaluation datasets."),
        ))
        if len(payloads) == 1 and payloads[0].get("layer_payloads"):
            layer_panels, dataset_panels = _breakdown_panels(payloads[0])
            outputs.extend(_render_grid(
                plt, layer_panels, metric=metric, ylabel=ylabel, style=style, output=output,
                stem=f"reconstruction-{metric}-by-layer", formats=formats, force=force,
                caption="One subplot per layer; each curve is averaged across datasets.",
            ))
            outputs.extend(_render_grid(
                plt, dataset_panels, metric=metric, ylabel=ylabel, style=style, output=output,
                stem=f"reconstruction-{metric}-by-dataset", formats=formats, force=force,
                caption="One subplot per dataset; each curve is averaged across layers.",
            ))
    return outputs


def _render_grid(
    plt: Any,
    panels: Sequence[tuple[str, list[dict[str, Any]]]],
    *,
    metric: str,
    ylabel: str,
    style: dict[str, tuple[str, str, str]],
    output: Path,
    stem: str,
    formats: Sequence[str],
    force: bool,
    caption: str,
) -> list[Path]:
    if not panels or any(not rows for _, rows in panels):
        raise ValueError("reconstruction experiment contains no completed rows")
    paths = [output / f"{stem}.{suffix}" for suffix in formats]
    caption_path = output / f"{stem}.txt"
    has_context_control = any(
        str(row.get("method", "")).startswith("sae_context_")
        for _, rows in panels
        for row in rows
    )
    for path in [*paths, caption_path]:
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    columns = min(3, len(panels))
    row_count = math.ceil(len(panels) / columns)
    with plt.rc_context({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        figure, axes = plt.subplots(
            row_count, columns, figsize=(2.75 * columns, 2.45 * row_count),
            sharey=True, squeeze=False,
        )
        handles: dict[str, Any] = {}
        for axis, (title, rows) in zip(axes.flat, panels, strict=False):
            for method in ("ica", "sae", "sae_context_64", "pca", "random"):
                points = _mean_curve(rows, method, metric)
                if not points:
                    continue
                label, color, marker = style[method]
                native = _native(rows, method, metric)
                endpoint = (
                    native
                    if native is not None
                    else points[-1] if method in {"ica", "pca", "random"} else None
                )
                marker_indices = [
                    index
                    for index, point in enumerate(points)
                    if endpoint is None or not _same_plot_point(point, endpoint)
                ]
                handles[method] = axis.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=color,
                    marker=marker,
                    markevery=marker_indices,
                    linestyle="--" if method.startswith("sae_context_") else "-",
                    linewidth=1.5,
                    markersize=4,
                    label=label,
                )[0]
                if endpoint is not None:
                    axis.scatter(*endpoint, marker="*", s=55, color=color, zorder=4)
            axis.set_xscale("log")
            axis.set_title(title)
            axis.set_xlabel("active directions per token")
            axis.grid(axis="y", alpha=0.25)
        for axis in list(axes.flat)[len(panels):]:
            axis.set_visible(False)
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        ordered = [
            handles[name]
            for name in ("ica", "sae", "sae_context_64", "pca", "random")
            if name in handles
        ]
        figure.legend(ordered, [item.get_label() for item in ordered], loc="upper center",
                      ncol=max(1, len(ordered)), frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        for path in paths:
            figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
    control_note = (
        " For GPT-2, the dashed SAE curve is a training-context control evaluated only "
        "at token positions 0–63; solid curves use the full 1,024-position context."
        if has_context_control
        else ""
    )
    caption_path.write_text(
        f"{caption} The plotted metric is {ylabel}. Stars mark each method's "
        "untruncated reconstruction: the complete basis for ICA, PCA, and Random, "
        "and native SAE sparsity."
        f"{control_note}\n",
        encoding="utf-8",
    )
    return [*paths, caption_path]


def _breakdown_panels(
    payload: dict[str, Any],
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[tuple[str, list[dict[str, Any]]]]]:
    from .reconstruction import _aggregate_layer

    layer_payloads = payload["layer_payloads"]
    layers = [(f"Layer {item['layer']}", item["rows"]) for item in layer_payloads]
    dataset_count = len(layer_payloads[0]["datasets"])
    datasets: list[tuple[str, list[dict[str, Any]]]] = []
    for index in range(dataset_count):
        rows: list[dict[str, Any]] = []
        config = layer_payloads[0]["datasets"][index]["dataset"]
        for item in layer_payloads:
            rows.extend(_aggregate_layer(int(item["layer"]), [item["datasets"][index]]))
        datasets.append((_dataset_title(config), rows))
    return layers, datasets


def _dataset_title(config: dict[str, Any]) -> str:
    """Return an unambiguous, compact title for a reconstruction dataset."""
    if config.get("repo_id") == "wikimedia/wikipedia" and config.get("domain"):
        return str(config["domain"])
    return str(config["repo_id"])


def _mean_curve(rows: list[dict[str, Any]], method: str, metric: str) -> list[tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("method") != method or row.get("k") == "native":
            continue
        grouped.setdefault(str(row["k"]), []).append(
            (float(row["effective_k"]), float(row[f"{metric}_mean"]))
        )
    requested_points = [
        (
            sum(x for x, _ in values) / len(values),
            sum(y for _, y in values) / len(values),
        )
        for _, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    ]
    points: list[tuple[float, float]] = []
    for x, y in requested_points:
        if points and math.isclose(x, points[-1][0], rel_tol=1e-5, abs_tol=1e-5):
            # Top-k SAEs cannot use more than their native active count. Requested
            # budgets above that count reconstruct the identical vector and belong
            # at one x coordinate, not several nearly vertical floating-point bins.
            points[-1] = ((points[-1][0] + x) / 2, (points[-1][1] + y) / 2)
        else:
            points.append((x, y))
    return points


def _native(rows: list[dict[str, Any]], method: str, metric: str) -> tuple[float, float] | None:
    selected = [row for row in rows if row.get("method") == method and row.get("k") == "native"]
    if not selected:
        return None
    return (
        sum(float(row["effective_k"]) for row in selected) / len(selected),
        sum(float(row[f"{metric}_mean"]) for row in selected) / len(selected),
    )


def _same_plot_point(
    point: tuple[float, float], endpoint: tuple[float, float]
) -> bool:
    """Return whether two markers would be visually indistinguishable on the plot."""
    return (
        math.isclose(point[0], endpoint[0], rel_tol=0.03, abs_tol=1e-6)
        and math.isclose(point[1], endpoint[1], rel_tol=0.005, abs_tol=0.005)
    )


def _load(path: Path) -> dict[str, Any]:
    result = path / "results.json"
    if not result.is_file():
        candidates = list(Path.cwd().glob("results/**/results.json"))
        ranked = sorted(
            candidates,
            key=lambda item: difflib.SequenceMatcher(
                None, str(path), str(item.parent.resolve())
            ).ratio(),
            reverse=True,
        )[:3]
        suggestion = ""
        if ranked:
            choices = "\n".join(f"  {item.parent}" for item in ranked)
            suggestion = f"\nAvailable result directories include:\n{choices}"
        raise FileNotFoundError(
            f"no reconstruction results at {result}. The input must match the "
            f"--output directory used by the experiment.{suggestion}"
        )
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["layer_payloads"] = [
        json.loads(layer.read_text(encoding="utf-8"))
        for layer in sorted((path / "layers").glob("layer_*.json"))
    ]
    return payload


def _titles(value: str | None, payloads: Sequence[dict[str, Any]]) -> list[str]:
    if value is not None:
        titles = [item.strip() for item in value.split(",")]
        if len(titles) != len(payloads):
            raise ValueError("--panel-titles must contain one title per experiment")
        return titles
    return [str(payload["experiment"]["model_id"]).split("/")[-1] for payload in payloads]
