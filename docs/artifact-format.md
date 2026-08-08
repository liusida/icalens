# ICA Lens artifact format

## Scope

Version 0.1 uses one Hugging Face Model repository for each base model. A
repository contains a manifest plus fitted ICA tensors for its available
layers. The format is identical when created with `save()`, stored in a local
directory, or uploaded with `push_to_hub()`.

The initial layout is:

```text
README.md
icalens.json
artifacts/
└── resid_post/
    ├── layer_00.safetensors
    ├── layer_01.safetensors
    └── layer_06.safetensors
```

The activation-site directory is retained even when a release contains only
`resid_post`. This leaves room for future sites such as `mlp_out` or
`attn_out` without changing the format.

## Manifest

`icalens.json` is the machine-readable source of truth. It must include:

- The artifact format name and integer format version.
- The base-model repository ID and exact fitted checkpoint revision.
- The activation site and layer-indexing convention.
- The hidden size and input preprocessing steps.
- The available layers, tensor filename, and component count for each layer.
- Essential fitting provenance, including algorithm, dtype, and random seed.

Example:

```json
{
  "format": "icalens",
  "format_version": 1,
  "base_model": {
    "repo_id": "openai-community/gpt2",
    "revision": "FULL_COMMIT_HASH"
  },
  "activation_site": "resid_post",
  "layer_indexing": "hidden_states",
  "hidden_size": 768,
  "input_preprocessing": {
    "row_normalization": "l2"
  },
  "layers": {
    "6": {
      "file": "artifacts/resid_post/layer_06.safetensors",
      "n_components": 768,
      "fitting": {
        "algorithm": "fastica",
        "implementation": "icalens.torch",
        "random_state": 0
      }
    }
  }
}
```

The precise base-model revision is part of compatibility validation. A model
name alone is not sufficient to identify the checkpoint that produced the
fitting activations.

## Layer tensors

Layer files use Safetensors rather than pickle. The public artifact contract is
defined in terms of the operations needed by ICA Lens, not the attribute names
of the fitting library.

The initial tensor names are:

```text
center             [hidden_size]
reading_matrix     [n_components, hidden_size]
writing_matrix     [hidden_size, n_components]
```

Conceptually, the operations are:

```python
processed = preprocess(activations)
scores = (processed - center) @ reading_matrix.T
reconstructed = scores @ writing_matrix.T + center
```

The final exporter and implementation must document the exact preprocessing
and equations. In particular, inverse transformation is approximate when input
preprocessing, such as row normalization, discards information.

## Loading and versioning

`ICALens.from_pretrained()` accepts either a Hugging Face Model repository ID
or a local directory. For a Hub repository, it loads `icalens.json` first and
may download individual layer files lazily. It accepts a `revision` argument so
users can pin a release tag or full commit hash.

Artifact releases should receive immutable tags such as `v0.1.0`. Readers must
reject unsupported `format_version` values with a clear compatibility error.

## Fitting and publishing

`ICALens.fit(activations, layer=...)` fits one layer at a time. Repeated calls
build a model-level collection; fitting an existing layer replaces that layer
only. All layers in a collection must have compatible base-model identity,
hidden size, activation site, and preprocessing settings.

Fitting must be reproducible when the same input, parameters, and
`random_state` are supplied. The manifest records the effective fitting
configuration and provenance. ICA Lens cannot verify that user-supplied
activations actually came from the declared model or layer, so this limitation
must be clear in the API and model card.

The v0.1 fitter is an internal PyTorch implementation of FastICA with
unit-variance whitening. It does not depend on scikit-learn or SciPy. NumPy
inputs default to CPU; callers can select a different fitting device. Mean,
covariance, fixed-point updates, and source variance are accumulated in
repeated batches, so the complete activation matrix need not be placed on the
GPU. The effective batch size, fitting device, and
`blockwise_multi_pass` memory strategy are recorded per layer.

`save(path)` writes the manifest and layer files locally. It must use an atomic
or staging-directory strategy so an interrupted save does not leave an
apparently valid partial artifact.

`push_to_hub(repo_id, ...)` is the explicit network operation for publishing
the saved representation to a Hugging Face Model repository. It should support
at least `private`, `token`, `revision`, and `commit_message` options. It must
not overwrite unrelated remote files silently. Official `liusida/icalens-*`
artifacts are generated with this same public fitting and publishing API rather
than a private exporter.

## Model card

The repository `README.md` is its Hugging Face model card. Its YAML metadata
should identify `icalens` as the library, name the base model and license, and
include interpretability and ICA tags. The body should document the activation
site, layer convention, fitting data and method, intended use, limitations,
package version, and paper citation.
