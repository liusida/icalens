"""Run the official SAE suffix ERF protocol, one resumable model/layer at a time."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from icalens._activation_dataset import ActivationDataset
from icalens.experiments.erf_suffix_sweep import _parse_rank_thresholds

ROOT = Path(__file__).resolve().parent
MODELS = {
    "gpt2": ("openai-community/gpt2", "gpt2-pile10k-1m", 12),
    "gemma2": ("google/gemma-2-2b", "gemma-2-2b-pile10k-1m", 26),
    "qwen9b": ("Qwen/Qwen3.5-9B-Base", "qwen3.5-9b-base-pile10k-1m", 32),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--layers", default="all", help="all or comma-separated zero-based layers")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/home/liusida/Expansion/research/ICA-data/icalens-activations"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sae-suffix-sweep-v1")
    parser.add_argument("--rank-thresholds", type=_parse_rank_thresholds, default=(1, 3, 5, 10, 15))
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--occurrences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache metadata and print jobs; no model loading",
    )
    args = parser.parse_args()
    jobs = []
    for label in dict.fromkeys(args.models):
        model, cache_name, n_layers = MODELS[label]
        cache_path = args.cache_root / cache_name
        cache = ActivationDataset(cache_path)
        layers = (
            list(range(n_layers))
            if args.layers == "all"
            else sorted(set(map(int, args.layers.split(","))))
        )
        if cache.model["repo_id"] != model or cache.sample_count != 1_000_000:
            raise ValueError(f"{label}: expected the official 1M-token cache")
        if not layers or any(
            layer not in cache.available_layers or not 0 <= layer < n_layers for layer in layers
        ):
            raise ValueError(f"{label}: missing/invalid layers {layers}")
        provenance = cache.provenance
        if (
            provenance["context_length"] != 1024
            or provenance["dataset"]["repo_id"] != "NeelNanda/pile-10k"
            or cache.manifest["activation_site"] != "resid_post"
        ):
            raise ValueError(f"{label}: incompatible cache protocol")
        for layer in layers:
            jobs.append(
                [
                    sys.executable,
                    "-m",
                    "icalens.experiments.erf_sae",
                    "--cache",
                    str(cache_path.resolve()),
                    "--layer",
                    str(layer),
                    "--output",
                    str((args.output / label / f"layer_{layer:02d}").resolve()),
                    "--rank-thresholds",
                    ",".join(map(str, args.rank_thresholds)),
                    "--features",
                    str(args.features),
                    "--occurrences",
                    str(args.occurrences),
                    "--seed",
                    str(args.seed),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "jobs": jobs,
                    "layer_count": len(jobs),
                    "gpt2_context_note": (
                        "1024-token evaluation; SAE trained with 64-token context. "
                        "Not context-matched."
                    ),
                },
                indent=2,
            )
        )
        return
    for job in jobs:
        # A fresh process releases each layer's GPU allocations and cached tensors.
        # The child owns validated atomic checkpoints, progress, and persistent logs.
        subprocess.run(job, check=True)


if __name__ == "__main__":
    main()
