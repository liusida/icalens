"""Summarize and plot the manual annotation confidence scores."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "annotations.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURE_PNG = FIGURES / "annotation-confidence.png"
FIGURE_PDF = FIGURES / "annotation-confidence.pdf"
KURTOSIS_FIGURE_PNG = FIGURES / "manual-confidence-vs-kurtosis.png"
KURTOSIS_FIGURE_PDF = FIGURES / "manual-confidence-vs-kurtosis.pdf"
WORD_CLOUD = FIGURES / "annotation-word-cloud.png"
SUMMARY = RESULTS / "annotation-summary.json"
MODEL_TITLES = {
    "gpt2": "GPT-2",
    "gemma-2-2b": "Gemma 2 2B",
    "qwen3.5-9b": "Qwen3.5 9B",
}
PROFILE_ROOTS = {
    "gpt2": ROOT.parent.parent / "local-icalens-models/official/icalens-gpt2-small-pile10k",
    "gemma-2-2b": ROOT.parent.parent
    / "local-icalens-models/official/icalens-gemma-2-2b-pile10k",
    "qwen3.5-9b": ROOT.parent.parent
    / "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k",
}
CONFIDENCE_COLORS = (
    "#E5E7EB",
    "#D6E0EF",
    "#B7CAE3",
    "#8FABD0",
    "#6687B5",
    "#3D5F99",
)
WORD_COLORS = ("#263B60", "#3D5F99", "#5879AC", "#7692BA", "#93A9C8")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def load_annotations() -> dict[str, list[dict[str, object]]]:
    payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotations.json must contain a model-to-annotations object")
    for model, rows in payload.items():
        if model not in MODEL_TITLES or not isinstance(rows, list):
            raise ValueError(f"unexpected annotation group: {model!r}")
        for row in rows:
            confidence = row.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, int):
                raise ValueError(f"{model} has a non-integer confidence: {confidence!r}")
            if confidence not in range(6):
                raise ValueError(f"{model} confidence must be between 0 and 5")
            if not str(row.get("label", "")).strip():
                raise ValueError(f"{model} has an empty label")
    return payload


def summarize(
    annotations: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    summary: dict[str, object] = {"confidence_scale": [0, 1, 2, 3, 4, 5], "groups": {}}
    for model, rows in annotations.items():
        values = np.asarray([int(row["confidence"]) for row in rows], dtype=np.int64)
        counts = Counter(values.tolist())
        summary["groups"][model] = {
            "n": int(values.size),
            "mean_confidence": float(values.mean()),
            "median_confidence": float(np.median(values)),
            "confidence_counts": {str(score): counts[score] for score in range(6)},
            "confidence_fractions": {
                str(score): counts[score] / int(values.size) for score in range(6)
            },
            "high_confidence_fraction": float(np.mean(values >= 4)),
            "unknown_fraction": float(np.mean(values == 0)),
        }
    return summary


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks with tied values assigned their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def load_kurtosis_rows(
    annotations: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, model_annotations in annotations.items():
        layers: dict[int, list[dict[str, object]]] = {}
        for annotation in model_annotations:
            layer = int(annotation["layer"])
            if layer not in layers:
                path = (
                    PROFILE_ROOTS[model]
                    / "component_profiles"
                    / "resid_post"
                    / f"layer_{layer:02d}.json.gz"
                )
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    layers[layer] = json.load(handle)["components"]
            component = int(annotation["component"])
            profile = layers[layer][component]
            if int(profile["component"]) != component:
                raise ValueError(f"component ordering mismatch in {model} layer {layer}")
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "component": component,
                    "confidence": int(annotation["confidence"]),
                    "excess_kurtosis": float(
                        profile["score_statistics"]["excess_kurtosis"]
                    ),
                }
            )
    return rows


def render_confidence_vs_kurtosis(rows: list[dict[str, object]]) -> dict[str, float]:
    kurtosis = np.asarray([row["excess_kurtosis"] for row in rows], dtype=np.float64)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    if np.any(kurtosis <= 0):
        raise ValueError("log-scaled kurtosis plot requires strictly positive values")

    x_edges = np.geomspace(0.5, 2000, 22)
    y_edges = np.arange(-0.5, 6.5, 1.0)
    counts, _, _ = np.histogram2d(kurtosis, confidence, bins=(x_edges, y_edges))
    maximum = int(counts.max())
    colors = plt.get_cmap("Blues", maximum + 1)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), colors.N)
    rho = spearman_correlation(kurtosis, confidence)

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.55, 2.75))
        mesh = axis.pcolormesh(x_edges, y_edges, counts.T, cmap=colors, norm=norm)
        axis.set_xscale("log")
        axis.set_xlim(x_edges[0], x_edges[-1])
        axis.set_ylim(y_edges[0], y_edges[-1])
        axis.set_yticks(range(6))
        axis.set_xlabel(r"Excess kurtosis $K$")
        axis.set_ylabel("Manual confidence")
        axis.grid(color="0.87", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.02,
            0.98,
            rf"$n={len(rows)},\ \rho_s={rho:.2f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.035, ticks=range(maximum + 1))
        colorbar.set_label("Components / bin")
        figure.subplots_adjust(left=0.18, right=0.87, bottom=0.20, top=0.96)
        figure.savefig(KURTOSIS_FIGURE_PNG, dpi=300)
        figure.savefig(KURTOSIS_FIGURE_PDF)
        plt.close(figure)
    return {"n": len(rows), "spearman_rho": rho}


def render(summary: dict[str, object]) -> None:
    keys = list(MODEL_TITLES)
    titles = [MODEL_TITLES[key] for key in keys]
    groups = summary["groups"]
    fractions = np.asarray(
        [[groups[key]["confidence_fractions"][str(score)] for score in range(6)] for key in keys]
    )

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(6.9, 2.45))
        y = np.arange(len(keys))
        left = np.zeros(len(keys))
        for score, color in enumerate(CONFIDENCE_COLORS):
            widths = fractions[:, score]
            bars = axis.barh(
                y,
                widths,
                left=left,
                height=0.58,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=str(score),
            )
            for bar, width in zip(bars, widths, strict=True):
                if width >= 0.075:
                    axis.text(
                        bar.get_x() + width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#172033",
                    )
            left += widths

        axis.set_yticks(y, titles)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Share of sampled components")
        axis.set_title("Annotation confidence", loc="left", fontweight="bold")
        axis.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
        axis.grid(axis="x", color="0.89", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            title="Confidence",
            ncol=6,
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1, 1.01),
            columnspacing=1.2,
            handlelength=1.5,
        )
        figure.subplots_adjust(left=0.20, right=0.96, bottom=0.23, top=0.76)
        figure.savefig(FIGURE_PNG, dpi=300)
        figure.savefig(FIGURE_PDF)
        plt.close(figure)


def render_word_cloud(annotations: dict[str, list[dict[str, object]]]) -> None:
    frequencies: Counter[str] = Counter()
    for rows in annotations.values():
        for row in rows:
            confidence = int(row["confidence"])
            if confidence == 0 or str(row["label"]).strip() == "?":
                continue
            words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", str(row["label"]).lower())
            frequencies.update(
                {word: confidence for word in words if len(word) > 1 and word not in STOP_WORDS}
            )

    width, height = 1200, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = font_manager.findfont("DejaVu Serif")
    title_font = ImageFont.truetype(font_path, 34)
    draw.text((32, 22), "Recurring concepts in manual labels", font=title_font, fill="#172033")
    placed: list[tuple[int, int, int, int]] = [(0, 0, width, 82)]
    randomizer = random.Random(0)
    selected = frequencies.most_common(45)
    maximum = selected[0][1]
    minimum = selected[-1][1]
    for rank, (word, frequency) in enumerate(selected):
        fraction = (frequency - minimum) / max(1, maximum - minimum)
        size = round(18 + 38 * math.sqrt(fraction))
        font = ImageFont.truetype(font_path, size)
        box = draw.textbbox((0, 0), word, font=font)
        word_width, word_height = box[2] - box[0], box[3] - box[1]
        phase = randomizer.random() * 2 * math.pi
        target: tuple[int, int, int, int] | None = None
        for step in range(2400):
            angle = phase + step * 0.31
            radius = 1.6 * math.sqrt(step)
            x = round(width / 2 + radius * 7.0 * math.cos(angle) - word_width / 2)
            y = round((height + 82) / 2 + radius * 3.2 * math.sin(angle) - word_height / 2)
            candidate = (x - 5, y - 4, x + word_width + 5, y + word_height + 4)
            if x < 24 or y < 90 or candidate[2] > width - 24 or candidate[3] > height - 20:
                continue
            if any(
                candidate[0] < other[2]
                and candidate[2] > other[0]
                and candidate[1] < other[3]
                and candidate[3] > other[1]
                for other in placed
            ):
                continue
            target = candidate
            draw.text(
                (x, y),
                word,
                font=font,
                fill=WORD_COLORS[rank % len(WORD_COLORS)],
            )
            placed.append(candidate)
            break
        if target is None:
            continue
    image.save(WORD_CLOUD, dpi=(240, 240))


def main() -> None:
    args = parse_args()
    existing = [
        path
        for path in (
            FIGURE_PNG,
            FIGURE_PDF,
            KURTOSIS_FIGURE_PNG,
            KURTOSIS_FIGURE_PDF,
            WORD_CLOUD,
            SUMMARY,
        )
        if path.exists()
    ]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to replace existing outputs: {joined}; pass --force")
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()
    summary = summarize(annotations)
    kurtosis_rows = load_kurtosis_rows(annotations)
    summary["confidence_vs_kurtosis"] = {
        **render_confidence_vs_kurtosis(kurtosis_rows),
        "components": kurtosis_rows,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render(summary)
    render_word_cloud(annotations)
    print(f"Wrote {FIGURE_PNG}")
    print(f"Wrote {FIGURE_PDF}")
    print(f"Wrote {KURTOSIS_FIGURE_PNG}")
    print(f"Wrote {KURTOSIS_FIGURE_PDF}")
    print(f"Wrote {WORD_CLOUD}")
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
