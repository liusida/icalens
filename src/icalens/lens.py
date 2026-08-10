"""The main ICA Lens API."""

from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from numpy.typing import NDArray

from ._arrays import transform_array
from ._artifact import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    LayerArtifact,
    layer_from_manifest,
    load_layer,
    read_manifest,
    save_directory,
)
from ._fastica import fit_fastica
from .exceptions import ArtifactError, NotFittedError


class ICALens:
    """A model-level collection of fitted layerwise ICA transformations."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
        model_type: Literal["base", "instruct"] = "base",
        base_model: str | None = None,
        base_model_revision: str | None = None,
        activation_site: str = "resid_post",
        layer_indexing: str = "hidden_states",
        row_normalize: bool = True,
        norm_eps: float = 1e-12,
    ) -> None:
        model_id = _resolve_compatibility_argument("model_id", model_id, "base_model", base_model)
        model_revision = _resolve_compatibility_argument(
            "model_revision", model_revision, "base_model_revision", base_model_revision
        )
        model_type = _validate_model_type(model_type)
        if not isinstance(activation_site, str) or not activation_site.strip():
            raise ValueError("activation_site must be a non-empty string")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", activation_site.strip()) is None:
            raise ValueError("activation_site may contain only letters, digits, '.', '_', and '-'")
        if not isinstance(layer_indexing, str) or not layer_indexing.strip():
            raise ValueError("layer_indexing must be a non-empty string")
        if not np.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be a finite positive number")
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_type = model_type
        self.activation_site = activation_site.strip()
        self.layer_indexing = layer_indexing.strip()
        self.row_normalize = bool(row_normalize)
        self.norm_eps = float(norm_eps)
        self._hidden_size: int | None = None
        self._layers: dict[int, LayerArtifact] = {}
        self._local_root: Path | None = None
        self._hub_source: dict[str, Any] | None = None

    @property
    def available_layers(self) -> tuple[int, ...]:
        """Sorted layer indices present in this lens."""
        return tuple(sorted(self._layers))

    @property
    def base_model(self) -> str:
        """Deprecated compatibility alias for :attr:`model_id`."""
        return self.model_id

    @property
    def base_model_revision(self) -> str:
        """Deprecated compatibility alias for :attr:`model_revision`."""
        return self.model_revision

    @property
    def hidden_size(self) -> int | None:
        """Activation width, or ``None`` before the first fit."""
        return self._hidden_size

    @property
    def metadata(self) -> dict[str, Any]:
        """A detached copy of the portable artifact manifest."""
        return copy.deepcopy(self._manifest())

    def fit(
        self,
        activations: Any,
        *,
        layer: int,
        n_components: int | None = None,
        algorithm: str = "parallel",
        fun: str = "logcosh",
        max_iter: int = 200,
        random_state: int | None = 0,
        progress: bool = False,
        device: str | torch.device | None = None,
        batch_size: int = 8192,
    ) -> ICALens:
        """Fit or replace the ICA transformation for one layer."""
        layer = _validate_layer(layer)
        values = _as_fit_tensor(activations)
        hidden_size = int(values.shape[1])
        if self._hidden_size is not None and hidden_size != self._hidden_size:
            raise ValueError(
                f"activation hidden size {hidden_size} does not match lens hidden size "
                f"{self._hidden_size}"
            )
        components = hidden_size if n_components is None else int(n_components)
        maximum = min(int(values.shape[0]) - 1, hidden_size)
        if maximum <= 0:
            raise ValueError("at least two activation samples are required after centering")
        if components <= 0 or components > maximum:
            raise ValueError(
                f"n_components must be between 1 and {maximum}, got {components}; "
                "centering limits the data rank to at most n_samples - 1"
            )
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        result = fit_fastica(
            values,
            n_components=components,
            algorithm=algorithm,
            fun=fun,
            max_iter=max_iter,
            random_state=random_state,
            progress=progress,
            device=device,
            batch_size=batch_size,
            row_normalize=self.row_normalize,
            norm_eps=self.norm_eps,
        )

        center = _to_numpy(result.center)
        reading = _to_numpy(result.components)
        writing = _to_numpy(result.mixing)
        filename = f"artifacts/{self.activation_site}/layer_{layer:02d}.safetensors"
        self._layers[layer] = LayerArtifact(
            layer=layer,
            file=filename,
            n_components=components,
            fitting={
                "algorithm": "fastica",
                "implementation": "icalens.torch",
                "implementation_version": "1",
                "torch_version": torch.__version__,
                "ica_algorithm": algorithm,
                "fun": fun,
                "whiten": "unit-variance",
                "whiten_solver": "eigh",
                "max_iter": int(max_iter),
                "stopping_criterion": "fixed_iterations",
                "random_state": random_state,
                "n_iter": result.n_iter,
                "n_samples": int(values.shape[0]),
                "input_dtype": str(values.dtype).removeprefix("torch."),
                "fit_device": str(result.center.device),
                "batch_size": int(batch_size),
                "memory_strategy": "blockwise_multi_pass",
                "stored_dtype": "float32",
                "component_id_convention": (
                    "row index; no post-fit sorting, sign canonicalization, or renumbering"
                ),
            },
            center=center,
            reading_matrix=reading,
            writing_matrix=writing,
        )
        self._hidden_size = hidden_size
        self._local_root = None
        self._hub_source = None
        return self

    def transform(self, activations: Any, *, layer: int) -> Any:
        """Map activations to signed ICA component scores."""
        artifact = self._get_layer(layer)
        self._validate_final_dimension(activations, self._hidden_size)
        assert artifact.center is not None and artifact.reading_matrix is not None
        return transform_array(
            activations,
            matrix=artifact.reading_matrix,
            offset=artifact.center,
            normalize=self.row_normalize,
            norm_eps=self.norm_eps,
        )

    def inverse_transform(self, scores: Any, *, layer: int) -> Any:
        """Approximately reconstruct preprocessed activations from scores."""
        artifact = self._get_layer(layer)
        self._validate_final_dimension(scores, artifact.n_components)
        assert artifact.center is not None and artifact.writing_matrix is not None
        return transform_array(
            scores,
            matrix=artifact.writing_matrix,
            offset=np.zeros(artifact.n_components, dtype=np.float32),
            normalize=False,
            norm_eps=self.norm_eps,
        ) + _array_offset(scores, artifact.center)

    def save(self, path: str | Path) -> Path:
        """Write this lens to a local artifact directory and return its path."""
        if not self._layers:
            raise NotFittedError("cannot save an ICA Lens with no fitted layers")
        for layer in self.available_layers:
            self._get_layer(layer)
        destination = Path(path).expanduser().resolve()
        save_directory(destination, self._manifest(), self._layers)
        return destination

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool | None = None,
        token: str | bool | None = None,
        revision: str = "main",
        commit_message: str = "Upload ICA Lens artifacts",
    ) -> str:
        """Create or update a Hugging Face Model repository."""
        if not repo_id or "/" not in repo_id:
            raise ValueError("repo_id must have the form 'owner/name'")
        with tempfile.TemporaryDirectory(prefix="icalens-upload-") as temporary:
            local_dir = self.save(Path(temporary) / "artifact")
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=local_dir,
                revision=revision,
                commit_message=commit_message,
            )
        return str(result)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> ICALens:
        """Load a lens manifest from a local directory or Hugging Face Hub."""
        candidate = Path(repo_id_or_path).expanduser()
        hub_source: dict[str, Any] | None = None
        if candidate.is_dir():
            root = candidate.resolve()
            manifest_path = root / MANIFEST_FILENAME
        else:
            repo_id = str(repo_id_or_path)
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=MANIFEST_FILENAME,
                    repo_type="model",
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            except Exception as error:
                raise ArtifactError(
                    f"could not download {repo_id}/{MANIFEST_FILENAME}: {error}"
                ) from error
            manifest_path = Path(downloaded)
            root = None
            hub_source = {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "token": token,
                "local_files_only": local_files_only,
                "force_download": force_download,
            }

        manifest = read_manifest(manifest_path)
        preprocessing = manifest["input_preprocessing"]
        if not isinstance(preprocessing, dict):
            raise ArtifactError("manifest input_preprocessing must be an object")
        normalization = preprocessing.get("row_normalization")
        if normalization not in ("l2", "none"):
            raise ArtifactError(f"unsupported row_normalization: {normalization!r}")
        format_version = int(manifest["format_version"])
        model = manifest["base_model"] if format_version == 1 else manifest["model"]
        lens = cls(
            model_id=str(model["repo_id"]),
            model_revision=str(model.get("revision") or "unknown"),
            model_type=_validate_model_type(str(model.get("type", "base"))),
            activation_site=str(manifest["activation_site"]),
            layer_indexing=str(manifest.get("layer_indexing", "hidden_states")),
            row_normalize=normalization == "l2",
            norm_eps=float(preprocessing.get("norm_eps", 1e-12)),
        )
        try:
            lens._hidden_size = int(manifest["hidden_size"])
        except (TypeError, ValueError) as error:
            raise ArtifactError("manifest hidden_size must be a positive integer") from error
        if lens._hidden_size <= 0:
            raise ArtifactError("manifest hidden_size must be a positive integer")
        lens._layers = {
            int(key): layer_from_manifest(key, value) for key, value in manifest["layers"].items()
        }
        lens._local_root = root
        lens._hub_source = hub_source
        return lens

    def _get_layer(self, layer: int) -> LayerArtifact:
        layer = _validate_layer(layer)
        if layer not in self._layers:
            available = ", ".join(map(str, self.available_layers)) or "none"
            raise NotFittedError(f"layer {layer} is unavailable; available layers: {available}")
        artifact = self._layers[layer]
        if not artifact.loaded:
            if self._hidden_size is None:
                raise ArtifactError("artifact has no hidden size")
            if self._local_root is not None:
                tensor_path = self._local_root / artifact.file
            elif self._hub_source is not None:
                source = self._hub_source
                try:
                    tensor_path = Path(
                        hf_hub_download(
                            repo_id=source["repo_id"],
                            filename=artifact.file,
                            repo_type="model",
                            revision=source["revision"],
                            cache_dir=source["cache_dir"],
                            token=source["token"],
                            local_files_only=source["local_files_only"],
                            force_download=source["force_download"],
                        )
                    )
                except Exception as error:
                    raise ArtifactError(f"could not download layer {layer}: {error}") from error
            else:
                raise ArtifactError(f"no source is available for unloaded layer {layer}")
            load_layer(tensor_path, artifact, self._hidden_size)
        return artifact

    def _manifest(self) -> dict[str, Any]:
        if self._hidden_size is None:
            hidden_size = 0
        else:
            hidden_size = self._hidden_size
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "model": {
                "repo_id": self.model_id,
                "revision": self.model_revision,
                "type": self.model_type,
            },
            "activation_site": self.activation_site,
            "layer_indexing": self.layer_indexing,
            "hidden_size": hidden_size,
            "input_preprocessing": {
                "row_normalization": "l2" if self.row_normalize else "none",
                "norm_eps": self.norm_eps,
            },
            "layers": {
                str(layer): {
                    "file": artifact.file,
                    "n_components": artifact.n_components,
                    "fitting": artifact.fitting,
                }
                for layer, artifact in sorted(self._layers.items())
            },
        }

    @staticmethod
    def _validate_final_dimension(values: Any, expected: int | None) -> None:
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError("input must have at least two dimensions")
        if expected is not None and int(shape[-1]) != expected:
            raise ValueError(f"input final dimension is {shape[-1]}, expected {expected}")


def _validate_layer(layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise TypeError("layer must be a non-negative integer")
    value = int(layer)
    if value < 0:
        raise ValueError("layer must be a non-negative integer")
    return value


def _resolve_compatibility_argument(
    name: str, value: str | None, legacy_name: str, legacy_value: str | None
) -> str:
    if value is not None and legacy_value is not None:
        raise ValueError(f"pass {name}, not both {name} and {legacy_name}")
    resolved = value if value is not None else legacy_value
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return resolved.strip()


def _validate_model_type(value: str) -> Literal["base", "instruct"]:
    if value == "base":
        return "base"
    if value == "instruct":
        return "instruct"
    raise ValueError("model_type must be 'base' or 'instruct'")


def _array_offset(reference: Any, offset: NDArray[np.float32]) -> Any:
    if type(reference).__module__.split(".", maxsplit=1)[0] == "torch":
        import torch

        return torch.as_tensor(offset, dtype=reference.dtype, device=reference.device)
    return offset


def _as_fit_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
    else:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("activations must have a floating-point dtype")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    if not tensor.is_floating_point():
        raise TypeError("activations must have a floating-point dtype")
    tensor = tensor.reshape(-1, tensor.shape[-1])
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("activations must contain only finite values")
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.float()
    return tensor


def _to_numpy(value: torch.Tensor) -> NDArray[np.float32]:
    return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())
