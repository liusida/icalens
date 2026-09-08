#!/usr/bin/env python3
"""Run the unchanged autointerpretability scorer over trajectory checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from icalens.cli._status import log
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty

from prepare import DEFAULT_OUTPUT, EVALUATED_CHECKPOINTS

ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATIONS = ROOT / "runs/evaluated"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_TINKER_EXPLAINER = "thinkingmachines/Inkling"
DEFAULT_TINKER_SIMULATOR = "Qwen/Qwen3.8-27B"


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
    if args.provider == "tinker" and args.model:
        raise ValueError("Tinker uses --explainer-model and --simulator-model, not --model")
    if args.provider == "openai" and (args.explainer_model or args.simulator_model):
        raise ValueError("OpenAI uses --model, not Tinker model-role options")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model = args.model or DEFAULT_OPENAI_MODEL if args.provider == "openai" else None
    explainer = (
        args.explainer_model or DEFAULT_TINKER_EXPLAINER
        if args.provider == "tinker"
        else None
    )
    simulator = (
        args.simulator_model or DEFAULT_TINKER_SIMULATOR
        if args.provider == "tinker"
        else None
    )
    preparation_manifests = {}
    for iteration in EVALUATED_CHECKPOINTS:
        path = source / f"iter-{iteration:03d}" / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"prepared checkpoint is missing: {path}")
        preparation_manifests[str(iteration)] = json.loads(path.read_text())["resolved"]
    resolved = {
        "format": "icalens.fastica_autointerpretability_evaluation",
        "format_version": 1,
        "input": str(source),
        "checkpoints": list(EVALUATED_CHECKPOINTS),
        "preparations": preparation_manifests,
        "provider": args.provider,
        "model": model,
        "explainer_model": explainer,
        "simulator_model": simulator,
        "methods": ["ica"],
        "n_features_per_layer": 50,
    }
    if args.dry_run:
        for iteration in EVALUATED_CHECKPOINTS:
            subprocess.run(
                _command(args, source, output, iteration, model, explainer, simulator),
                check=True,
            )
        return
    provenance = source_provenance()
    run = ResumableRun.open(
        output=output,
        filename="launcher-run.json",
        resolved=resolved,
        source=provenance,
        status="evaluating",
    )
    completed = {
        iteration
        for iteration in EVALUATED_CHECKPOINTS
        if _is_complete(output / f"iter-{iteration:03d}", source, iteration, args.provider)
    }
    try:
        with ExperimentDisplay(
            output=output / "logs",
            title=f"ICA Lens · autointerpretability convergence · {args.provider}",
            completed=len(completed),
            total=len(EVALUATED_CHECKPOINTS),
            completed_unit_ids=completed,
            source_dirty=provenance.get("dirty"),
            unit_label="fitting checkpoints",
            detail_filename="evaluation-detail.log",
        ) as display:
            warn_if_dirty(provenance)
            for iteration in EVALUATED_CHECKPOINTS:
                if iteration in completed:
                    log(f"Reused completed autointerpretability iteration {iteration}.")
                    continue
                display.phase("Evaluating checkpoint", iteration=iteration, provider=args.provider)
                log(f"Starting autointerpretability iteration {iteration}.")
                command = _command(args, source, output, iteration, model, explainer, simulator)
                child = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert child.stdout is not None
                for line in child.stdout:
                    print(line, end="")
                status = child.wait()
                if status:
                    raise subprocess.CalledProcessError(status, command)
                if not _is_complete(
                    output / f"iter-{iteration:03d}", source, iteration, args.provider
                ):
                    raise ValueError(f"child did not produce a valid iteration {iteration}")
                display.complete_unit(iteration, refresh=True)
                log(f"Completed autointerpretability iteration {iteration}.")
            run.set_status("complete", complete=True)
            log(f"Complete: {len(EVALUATED_CHECKPOINTS)} fitting checkpoints.")
    except BaseException:
        run.set_status("interrupted")
        raise


def _command(args, source, output, iteration, model, explainer, simulator):
    command = [
        "icalens", "experiment", "autointerpretability", "evaluate",
        "--input", str(source / f"iter-{iteration:03d}"),
        "--output", str(output / f"iter-{iteration:03d}"),
        "--methods", "ica", "--provider", args.provider,
        "--env-file", str(args.env_file),
    ]
    if model:
        command.extend(("--model", model))
    if explainer:
        command.extend(("--explainer-model", explainer))
    if simulator:
        command.extend(("--simulator-model", simulator))
    if args.dry_run:
        command.append("--dry-run")
    return command


def _is_complete(directory: Path, source: Path, iteration: int, provider: str) -> bool:
    path = directory / "run.json"
    try:
        state = json.loads(path.read_text())
        resolved = state["resolved"]
        return (
            state.get("status") == "complete"
            and Path(resolved["preparation"]).resolve()
            == (source / f"iter-{iteration:03d}").resolve()
            and resolved.get("methods") == ["ica"]
            and state.get("evaluation", {}).get("provider") == provider
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    main()
