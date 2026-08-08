# ICA Lens API notes

## Initial scope

Version 0.1 should focus on fitting ICA Lenses from activations supplied by
researchers, saving and sharing the resulting artifacts, and applying published
artifacts. Higher-level text analysis, model execution, and rich result objects
can be added in a future version.

```bash
pip install icalens
```

```python
from icalens import ICALens

lens = ICALens.from_pretrained("liusida/icalens-gpt2-small")

scores = lens.transform(
    activations,
    layer=6,
)

reconstructed_activations = lens.inverse_transform(
    scores,
    layer=6,
)
```

Users can fit a lens one layer at a time:

```python
from icalens import ICALens

lens = ICALens(
    base_model="openai-community/gpt2",
    base_model_revision="FULL_COMMIT_HASH",
    activation_site="resid_post",
)

lens.fit(
    activations,
    layer=6,
    random_state=0,
    device="cuda",
    batch_size=8192,
    progress=True,
)

lens.save("./icalens-gpt2-small")
lens.push_to_hub("username/icalens-gpt2-small")
```

Calling `fit()` again with another layer adds or replaces that layer in the
same model-level collection. `fit()` returns `self`. The input is an activation
array whose final dimension is the base model's hidden size; leading dimensions
are treated as samples.

Fitting is blockwise and multi-pass. `device` selects where whitening and
FastICA calculations run, while the activation tensor may remain in CPU
memory. `batch_size` bounds the number of activation rows moved to the fitting
device at once. GPU memory therefore scales primarily with batch size rather
than the total token count. The input must remain available for repeated passes
through mean estimation, covariance accumulation, fixed-point iterations, and
source scaling.

`save()` writes the standard artifact layout to a local directory. Network
upload is a separate, explicit operation: `push_to_hub()` creates or updates a
Hugging Face Model repository using that same layout. The official
`liusida/icalens-*` artifacts should be produced through these public methods.

The argument to `ICALens.from_pretrained()` is a Hugging Face Model repository
ID. Each repository contains the fitted ICA artifacts for one base model and
may contain multiple layers. `transform()` selects the layer to apply.

`from_pretrained()` should load the repository manifest first. Layer tensor
files may then be downloaded lazily when a layer is used. Hugging Face's cache
is used by default. The same method should accept a local directory with the
same artifact structure.

A `revision` argument should allow callers to pin a Hub tag or commit:

```python
lens = ICALens.from_pretrained(
    "liusida/icalens-gpt2-small",
    revision="v0.1.0",
)
```

The v0.1 interface should support:

- Fitting an ICA Lens for one or more layers from user-supplied activations.
- Saving fitted artifacts in the standard local format.
- Creating or updating a user-owned Hugging Face Model repository.
- Loading and caching ICA Lens artifacts from Hugging Face.
- Inspecting available layers, activation sites, base-model identity, and fitting metadata.
- Validating activation shape and artifact compatibility.
- Transforming hidden-state activations into ICA component scores.
- Approximately mapping component scores back into activation space.
- Loading the same artifact format from either the Hugging Face Hub or a local
  directory.

The caller is responsible for loading the language model and capturing the correct activations. The following are intentionally deferred:

- `lens.analyze(text, ...)` and `AnalysisResult`.
- Automatic model loading, tokenization, and activation capture.
- Visualization and explorer integration.
- In-model interventions and paper-reproduction workflows.

The fitting API does not capture activations or infer whether they came from
the declared model, revision, layer, or activation site. The caller supplies
that provenance, and ICA Lens records it in the manifest.

Fitting uses the package's internal PyTorch FastICA implementation. NumPy
inputs fit on CPU, while PyTorch inputs fit on their current device. ICA Lens
does not depend on scikit-learn or SciPy.

Future versions can build the high-level API on the same foundation:

```python
result = lens.analyze(
    "She deposited the check at the bank.",
    layer=6,
)
```
