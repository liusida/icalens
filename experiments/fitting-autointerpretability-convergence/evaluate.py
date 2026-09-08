#!/usr/bin/env python3
"""Run the unchanged autointerpretability scorer over trajectory checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare import DEFAULT_OUTPUT, EVALUATED_CHECKPOINTS

ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATIONS = ROOT / "runs/evaluated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--provider", choices=("tinker", "openai"), default="tinker")
    parser.add_argument("--model")
    parser.add_argument("--explainer-model")
    parser.add_argument("--simulator-model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for iteration in EVALUATED_CHECKPOINTS:
        print(f"START autointerpretability iter-{iteration:03d}", flush=True)
        command = [
            "icalens",
            "experiment",
            "autointerpretability",
            "evaluate",
            "--input",
            str(source / f"iter-{iteration:03d}"),
            "--output",
            str(output / f"iter-{iteration:03d}"),
            "--methods",
            "ica",
            "--provider",
            args.provider,
            "--env-file",
            str(args.env_file),
        ]
        if args.model:
            command.extend(("--model", args.model))
        if args.explainer_model:
            command.extend(("--explainer-model", args.explainer_model))
        if args.simulator_model:
            command.extend(("--simulator-model", args.simulator_model))
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)
        print(f"PASS autointerpretability iter-{iteration:03d}", flush=True)
    print(f"PASS all {len(EVALUATED_CHECKPOINTS)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
