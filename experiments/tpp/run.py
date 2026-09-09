"""Prepare and run the five-representation SAEBench TPP experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from icalens import ICALens
from icalens.experiments._display import ExperimentDisplay
from icalens.experiments._run import ResumableRun, atomic_write_json
from icalens.experiments._saebench_environment import prepare_backend, resolve_backend
from icalens.experiments._source_provenance import source_provenance, warn_if_dirty
from icalens.experiments.saebench_sparse_probing import (
    _parse_layers,
    _resolve_baselines,
    _write_layer_snapshot,
)

METHODS = ("ica", "unfitted_ica", "sae", "untrained_sae_matched_l0", "pca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--n-values", default=None, help="Comma-separated feature budgets.")
    parser.add_argument("--saebench-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.time()
    args = parse_args()
    source = source_provenance()
    warn_if_dirty(source)
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layers, lens.available_layers)
    backend = resolve_backend(lens.model_id)
    baselines = _resolve_baselines(lens.model_id, "sae,pca")
    settings = settings_for(args.preset, args.n_values)
    output = args.output.expanduser().resolve()
    fitting_seeds = {
        str(layer): int(lens._get_layer(layer).fitting["random_state"]) for layer in layers
    }
    layer_fingerprints = {str(layer): layer_fingerprint(lens, layer) for layer in layers}
    config = {
        "schema_version": 2,
        "experiment": "tpp-five-representation",
        "lens": str(args.lens),
        "model_id": lens.model_id,
        "model_revision": lens.model_revision,
        "layers": layers,
        "methods": list(METHODS),
        "method_definition_version": 2,
        "settings": settings,
        "saebench_backend": asdict(backend),
        "baseline_definitions": baselines,
        "fitting_seed_by_layer": fitting_seeds,
        "lens_layer_sha256": layer_fingerprints,
        "untrained_sae_definition": "tied-random-decoder-matched-l0",
        "unfitted_ica_definition": "seeded-fastica-symmetric-decorrelation-after-whitening",
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    validate_or_write(config_path, config)
    run = ResumableRun.open(
        output=output,
        resolved=config,
        source=source,
        status="running",
    )
    completed_layers = {
        layer
        for layer in layers
        if valid_layer_result(
            output / "layers" / f"layer_{layer:02d}" / "result.json",
            settings=settings,
        )
    }
    if len(completed_layers) == len(layers):
        run.set_status("complete", complete=True)
        print(f"All {len(layers)} layer(s) are already complete.")
        return
    prepared = prepare_backend(
        backend,
        cache_dir=args.cache_dir,
        saebench_path=args.saebench_path,
        refresh=False,
    )
    display = ExperimentDisplay(
        output=output / "logs",
        title="ICA Lens · targeted probe perturbation",
        completed=len(completed_layers),
        total=len(layers),
        source_dirty=bool(source.get("dirty")),
        unit_label="layers",
        started_at=started_at,
        completed_unit_ids=completed_layers,
    )
    with display:
        for layer in layers:
            if layer in completed_layers:
                continue
            display.phase("Evaluating five representations", Layer=layer)
            layer_dir = output / "layers" / f"layer_{layer:02d}"
            snapshot = _write_layer_snapshot(
                lens,
                layer=layer,
                output=output / "checkpoints" / f"layer_{layer:02d}",
                saebench_model_name=backend.saebench_model_name,
                baselines=baselines,
            )
            snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
            snapshot_payload["fitting_seed"] = fitting_seeds[str(layer)]
            atomic_write_json(snapshot, snapshot_payload)
            command = [
                str(prepared.python),
                str(Path(__file__).with_name("worker.py")),
                "--saebench-root",
                str(prepared.root),
                "--snapshot",
                str(snapshot),
                "--config",
                str(config_path),
                "--output",
                str(layer_dir),
            ]
            print("RUN " + " ".join(command), flush=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            display.complete_unit(layer, refresh=True)
    run.set_status("complete", complete=True)


def valid_layer_result(path: Path, *, settings: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("method_definition_version") != 2:
            return False
        methods = payload["methods"]
        expected = {f"{name}_custom_sae" for name in METHODS}
        if set(methods) != expected:
            return False
        for result in methods.values():
            config = result["eval_config"]
            if list(config["dataset_names"]) != list(settings["datasets"]):
                return False
            if list(config["n_values"]) != list(settings["n_values"]):
                return False
            if int(config["train_set_size"]) != int(settings["train_size"]):
                return False
            if int(config["test_set_size"]) != int(settings["test_size"]):
                return False
            metrics = result["eval_result_metrics"]["tpp_metrics"]
            for budget in settings["n_values"]:
                for suffix in ("total_metric", "intended_diff_only", "unintended_diff_only"):
                    value = metrics[f"tpp_threshold_{int(budget)}_{suffix}"]
                    if value is None or not isinstance(value, (int, float)):
                        return False
        return set(payload["feature_configs"]) == set(METHODS)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def layer_fingerprint(lens: ICALens, layer: int) -> str:
    artifact = lens._get_layer(layer)
    digest = hashlib.sha256()
    for value in (artifact.center, artifact.reading_matrix, artifact.writing_matrix):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def settings_for(preset: str, n_values: str | None) -> dict[str, object]:
    if preset == "smoke":
        settings: dict[str, object] = {
            "datasets": ["canrager/amazon_reviews_mcauley_1and5"],
            "n_values": [1, 2, 5, 10],
            "train_size": 200,
            "test_size": 100,
            "context_length": 128,
            "probe_epochs": 4,
            "llm_batch_size": 1,
            "sae_batch_size": 64,
            "llm_dtype": "float32",
            "random_seed": 42,
        }
    else:
        settings = {
            "datasets": [
                "LabHC/bias_in_bios_class_set1",
                "canrager/amazon_reviews_mcauley_1and5",
            ],
            "n_values": [1, 2, 5, 10, 20, 50, 100],
            "train_size": 2000,
            "test_size": 1000,
            "context_length": 128,
            "probe_epochs": 20,
            "llm_batch_size": 1,
            "sae_batch_size": 125,
            "llm_dtype": "float32",
            "random_seed": 42,
        }
    if n_values:
        settings["n_values"] = [int(value) for value in n_values.split(",")]
    return settings


def validate_or_write(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"incompatible existing run configuration: {path}")
        return
    atomic_write_json(path, payload)


if __name__ == "__main__":
    main()
