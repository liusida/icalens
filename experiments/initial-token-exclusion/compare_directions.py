"""Match and plot reading directions from the released and initial-token-exclusion lenses."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "local-icalens-models/official/icalens-gpt2-small-pile10k"
DEFAULT_CANDIDATE = Path(__file__).resolve().parent / "runs/gpt2-no-framing-skip-2"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/gpt2-layer-6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = _load_layer(args.reference, args.layer)
    candidate = _load_layer(args.candidate, args.layer)
    if reference["reading"].shape != candidate["reading"].shape:
        raise ValueError(
            f"reading-matrix shapes differ: {reference['reading'].shape} and "
            f"{candidate['reading'].shape}"
        )

    cosine = np.abs(reference["reading"] @ candidate["reading"].T)
    assignment = _maximum_weight_assignment(cosine)
    matched = cosine[np.arange(len(assignment)), assignment]
    reordered = cosine[:, assignment]
    rows = [
        {
            "reference_component": index,
            "candidate_component": int(assignment[index]),
            "absolute_cosine_similarity": float(matched[index]),
            "reference_strength": float(reference["strengths"][index]),
            "candidate_strength": float(candidate["strengths"][assignment[index]]),
        }
        for index in range(len(assignment))
    ]
    summary = {
        "comparison": "optimal one-to-one reading-direction matching",
        "reference": str(Path(args.reference).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "layer": args.layer,
        "component_count": len(assignment),
        "mean": float(np.mean(matched)),
        "median": float(np.median(matched)),
        "p05": float(np.quantile(matched, 0.05)),
        "minimum": float(np.min(matched)),
        "fraction_ge_0.70": float(np.mean(matched >= 0.70)),
        "fraction_ge_0.90": float(np.mean(matched >= 0.90)),
        "fraction_ge_0.95": float(np.mean(matched >= 0.95)),
    }

    output = Path(args.output).resolve()
    paths = {
        "csv": output / "matched-components.csv",
        "json": output / "summary.json",
        "png": output / "matched-reading-directions.png",
        "pdf": output / "matched-reading-directions.pdf",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"refusing to overwrite {existing}; pass --force")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["csv"], rows)
    paths["json"].write_text(json.dumps(summary, indent=2) + "\n")
    _plot(paths["png"], paths["pdf"], matched, reordered, layer=args.layer)
    print(json.dumps(summary, indent=2))


def _load_layer(root: Path, layer: int) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = json.loads((root / "icalens.json").read_text())
    record = manifest.get("layers", {}).get(str(layer))
    if not isinstance(record, dict):
        raise ValueError(f"{root} has no fitted layer {layer}")
    fitting = record.get("fitting")
    if not isinstance(fitting, dict):
        raise ValueError(f"{root} layer {layer} has no fitting metadata")
    strengths = np.asarray(fitting.get("component_strengths"), dtype=np.float64)
    reading = np.asarray(load_file(root / record["file"])["reading_matrix"], dtype=np.float64)
    if reading.ndim != 2 or strengths.shape != (reading.shape[0],):
        raise ValueError(f"invalid layer {layer} data in {root}")
    norms = np.linalg.norm(reading, axis=1, keepdims=True)
    if not np.isfinite(reading).all() or np.any(norms == 0):
        raise ValueError(f"invalid reading directions in {root}")
    return {"reading": reading / norms, "strengths": strengths}


def _maximum_weight_assignment(weights: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray:
    """Return the maximizing column for every row using the Hungarian algorithm."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("assignment weights must be square")
    size = weights.shape[0]
    costs = np.max(weights) - weights
    row_potential = np.zeros(size + 1)
    column_potential = np.zeros(size + 1)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1, candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    png: Path,
    pdf: Path,
    matched: np.ndarray[Any, np.dtype[np.float64]],
    reordered: np.ndarray[Any, np.dtype[np.float64]],
    *,
    layer: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="icalens-mpl-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
            axes[0].hist(matched, bins=np.linspace(0, 1, 41), color="#315b7d")
            axes[0].axvline(np.median(matched), color="#c44e52", linewidth=1)
            axes[0].set(xlabel="Matched |cosine|", ylabel="Components", xlim=(0, 1))

            axes[1].scatter(np.arange(len(matched)), matched, s=5, alpha=0.65, color="#315b7d")
            axes[1].set(
                xlabel="Released component strength rank",
                ylabel="Matched |cosine|",
                ylim=(0, 1.01),
            )

            image = axes[2].imshow(
                reordered,
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=1,
                cmap="magma",
                interpolation="nearest",
            )
            axes[2].set(
                xlabel="New components, reordered by match",
                ylabel="Released components",
            )
            figure.colorbar(image, ax=axes[2], label="Absolute cosine")
            figure.suptitle(f"GPT-2 layer {layer}: released vs no-framing, skip-2 fit")
            figure.savefig(png, dpi=200)
            figure.savefig(pdf)
            plt.close(figure)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


if __name__ == "__main__":
    main()
