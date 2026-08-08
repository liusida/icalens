"""Portable ICA Lens manifest and tensor persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file

from .exceptions import ArtifactError

FORMAT_NAME = "icalens"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "icalens.json"


@dataclass
class LayerArtifact:
    """Parameters and provenance for one fitted layer."""

    layer: int
    file: str
    n_components: int
    fitting: dict[str, Any]
    center: NDArray[np.float32] | None = None
    reading_matrix: NDArray[np.float32] | None = None
    writing_matrix: NDArray[np.float32] | None = None

    @property
    def loaded(self) -> bool:
        return self.center is not None


def parse_manifest(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ArtifactError("icalens.json must contain a JSON object")
    if data.get("format") != FORMAT_NAME:
        raise ArtifactError(f"unsupported artifact format: {data.get('format')!r}")
    if data.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(
            f"unsupported artifact format version: {data.get('format_version')!r}; "
            f"this package supports version {FORMAT_VERSION}"
        )
    required = ("base_model", "activation_site", "hidden_size", "input_preprocessing", "layers")
    missing = [key for key in required if key not in data]
    if missing:
        raise ArtifactError(f"manifest is missing required fields: {', '.join(missing)}")
    if not isinstance(data["base_model"], dict) or not data["base_model"].get("repo_id"):
        raise ArtifactError("manifest base_model.repo_id must be a non-empty string")
    if not isinstance(data["layers"], dict):
        raise ArtifactError("manifest layers must be an object")
    return data


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        return parse_manifest(json.loads(path.read_text(encoding="utf-8")))
    except ArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read manifest {path}: {error}") from error


def layer_from_manifest(layer_key: str, data: Any) -> LayerArtifact:
    try:
        layer = int(layer_key)
        if not isinstance(data, dict):
            raise TypeError
        filename = str(data["file"])
        n_components = int(data["n_components"])
        fitting = dict(data["fitting"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"invalid manifest entry for layer {layer_key!r}") from error
    if layer < 0 or n_components <= 0 or not filename:
        raise ArtifactError(f"invalid manifest entry for layer {layer_key!r}")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactError(f"unsafe tensor path for layer {layer}: {filename!r}")
    return LayerArtifact(layer=layer, file=filename, n_components=n_components, fitting=fitting)


def load_layer(path: Path, artifact: LayerArtifact, hidden_size: int) -> None:
    try:
        tensors = load_file(path)
        center = _tensor(tensors, "center", (hidden_size,))
        reading = _tensor(tensors, "reading_matrix", (artifact.n_components, hidden_size))
        writing = _tensor(tensors, "writing_matrix", (hidden_size, artifact.n_components))
    except ArtifactError:
        raise
    except Exception as error:
        raise ArtifactError(f"could not load layer artifact {path}: {error}") from error
    artifact.center = center
    artifact.reading_matrix = reading
    artifact.writing_matrix = writing


def save_directory(path: Path, manifest: dict[str, Any], layers: dict[int, LayerArtifact]) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ArtifactError(f"save destination exists and is not a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    backup: Path | None = None
    try:
        for artifact in layers.values():
            if not artifact.loaded:
                raise ArtifactError(f"layer {artifact.layer} tensors are not loaded")
            assert artifact.center is not None
            assert artifact.reading_matrix is not None
            assert artifact.writing_matrix is not None
            tensor_path = stage / artifact.file
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "center": artifact.center,
                    "reading_matrix": artifact.reading_matrix,
                    "writing_matrix": artifact.writing_matrix,
                },
                tensor_path,
            )
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "README.md").write_text(_model_card(manifest), encoding="utf-8")
        if path.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{path.name}-backup-", dir=path.parent))
            backup.rmdir()
            os.replace(path, backup)
        os.replace(stage, path)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists() and not path.exists():
            os.replace(backup, path)
        raise


def _tensor(
    tensors: dict[str, NDArray[Any]], name: str, shape: tuple[int, ...]
) -> NDArray[np.float32]:
    if name not in tensors:
        raise ArtifactError(f"layer artifact is missing tensor {name!r}")
    value = np.asarray(tensors[name], dtype=np.float32)
    if value.shape != shape:
        raise ArtifactError(f"tensor {name!r} has shape {value.shape}, expected {shape}")
    if not np.all(np.isfinite(value)):
        raise ArtifactError(f"tensor {name!r} contains non-finite values")
    return np.ascontiguousarray(value)


def _model_card(manifest: dict[str, Any]) -> str:
    model = manifest["base_model"]["repo_id"]
    site = manifest["activation_site"]
    layers = ", ".join(str(layer) for layer in sorted(map(int, manifest["layers"])))
    return f"""---
library_name: icalens
base_model: {model}
tags:
- interpretability
- independent-component-analysis
- activations
---

# ICA Lens for {model}

This repository contains an ICA Lens fitted on `{site}` activations from
`{model}`. Available layers: {layers}.

```python
from icalens import ICALens

lens = ICALens.from_pretrained(\"REPOSITORY_ID\")
scores = lens.transform(activations, layer={min(map(int, manifest["layers"]))})
```

The caller is responsible for capturing activations from the base-model
revision and activation site recorded in `icalens.json`.
"""
