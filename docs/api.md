# Python API

The public package interface consists of `ICALens`, `CaptureResult`,
`AnalysisResult`, and the package exceptions.

```python
from icalens import ICALens
```

## Create or load a lens

### `ICALens(...)`

Create an empty lens before fitting activation tensors:

```python
lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `model_id` | `str`, required | Hugging Face repository of the language model being analyzed |
| `model_revision` | `str \| None = None` | Exact language-model revision; resolved on first capture if omitted |
| `model_type` | `"base" \| "instruct" = "base"` | Whether the checkpoint is a base or instruction/chat model |
| `activation_site` | `str = "resid_post"` | Named activation location; the built-in capture path supports post-block residual streams |
| `layer_indexing` | `str = "hidden_states"` | Convention recorded in the artifact for interpreting layer numbers |
| `row_normalize` | `bool = True` | Apply per-token L2 normalization before centering and ICA |
| `norm_eps` | `float = 1e-12` | Numerical floor used during normalization |
| **Returns** | `ICALens` | A new, unfitted lens |

`model_revision` identifies the analyzed language-model weights. It is
different from the `revision` argument of `from_pretrained()`, which selects a
branch, tag, or commit of the **ICA Lens repository**.

### `ICALens.from_pretrained(...)`

Load a published lens:

```python
lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
```

Or load a local artifact directory:

```python
lens = ICALens.from_pretrained("icalens-output/my-icalens")
```

```python
ICALens.from_pretrained(
    repo_id_or_path,
    *,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
    force_download=False,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `repo_id_or_path` | `str \| Path`, required | Hugging Face Model repository or local artifact directory |
| `revision` | `str \| None = None` | Branch, tag, or commit of the ICA Lens repository |
| `cache_dir` | `str \| Path \| None = None` | Optional Hugging Face cache directory |
| `token` | `str \| bool \| None = None` | Hugging Face authentication setting |
| `local_files_only` | `bool = False` | Refuse network access and use cached files only |
| `force_download` | `bool = False` | Download the manifest and lazy layer files again |
| **Returns** | `ICALens` | Lens initialized from the artifact manifest |

For a Hub repository, only `icalens.json` is fetched initially. The tensor file
for a fitted layer is downloaded when that layer is first used. `revision`
selects the lens artifact revision; `token` accepts a Hugging Face token or the
standard `huggingface_hub` boolean token setting.

## Analyze model inputs

### `analyze(...)`

Capture activations and immediately transform them into ICA scores and energy
shares:

```python
result = lens.analyze(
    "She deposited the check at the bank.",
    layer=6,
)
```

For an instruction model, pass a completed conversation:

```python
result = lens.analyze(
    [
        {"role": "user", "content": "Name an interesting science."},
        {"role": "assistant", "content": "Physics."},
    ],
    layer=16,
    token_scope="all",
)
```

```python
lens.analyze(
    inputs,
    *,
    layer,
    model=None,
    tokenizer=None,
    token_scope="all",
    context_length=None,
    device="auto",
) -> AnalysisResult
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`, required | Raw text or a completed `{role, content}` conversation |
| `layer` | `int`, required | Zero-based transformer-block index stored in the lens |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | Conversation positions to return; raw text always returns all encoded tokens |
| `context_length` | `int \| None = None` | Optional maximum encoded input length |
| `device` | `str \| torch.device \| None = "auto"` | Model device; automatic mode prefers CUDA and otherwise uses CPU |
| **Returns** | `AnalysisResult` | Tokens, activations, signed scores, and energy shares |

The first call lazily loads the language model and tokenizer. Subsequent calls
on the same lens reuse them.

### `generate(...)`

Generate a continuation, optionally clamping one signed ICA coordinate:

```python
baseline = lens.generate(messages, max_new_tokens=16)

steered = lens.generate(
    messages,
    layer=5,
    clamp=(188, -20.0),
    max_new_tokens=16,
)
```

```python
lens.generate(
    prompt,
    *,
    layer=None,
    clamp=None,
    max_new_tokens=64,
    device="auto",
    model=None,
    tokenizer=None,
    **generation_kwargs,
) -> str
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `prompt` | `str \| list[dict[str, str]]`, required | Raw prompt or chat messages; chat templates are applied automatically |
| `layer` | `int \| None = None` | Zero-based residual-stream layer to edit; required with `clamp` |
| `clamp` | `tuple[int, float] \| Mapping[int, float] \| None = None` | One `(component, target_score)` pair or a mapping of simultaneous clamps, applied at every processed token position and generation step |
| `max_new_tokens` | `int = 64` | Maximum number of continuation tokens |
| `device` | `str \| torch.device \| None = "auto"` | Model device; automatic mode prefers CUDA and otherwise uses CPU |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `**generation_kwargs` | keyword arguments | Additional arguments forwarded to `model.generate()` |
| **Returns** | `str` | Decoded continuation only, without the prompt |

Without `clamp`, this is ordinary generation. With `clamp`, ICA Lens edits the
`resid_post` activation at the selected layer, then restores each activation's
original norm before returning it to the model. Greedy decoding is the default;
pass standard generation arguments to choose another decoding strategy. The
language model is loaded lazily and reused by later calls on the same lens.

### `capture(...)`

Use `capture()` when you need aligned activations without transforming them:

```python
captured = lens.capture("She deposited the check.", layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`, required | Raw text or a completed `{role, content}` conversation |
| `layer` | `int`, required | Zero-based transformer-block index to capture |
| `model` | `torch.nn.Module \| None = None` | Optional caller-supplied language model |
| `tokenizer` | tokenizer or `None` | Optional caller-supplied tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | Conversation positions included in the result |
| `context_length` | `int \| None = None` | Optional maximum encoded input length |
| `device` | `str \| torch.device \| None = "auto"` | Model device |
| **Returns** | `CaptureResult` | Tokens, positions, IDs, and aligned activations |

### `unload_model()`

```python
lens.unload_model()
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| **Returns** | `None` | Releases the model and tokenizer cached by `capture()`, `analyze()`, or `generate()` |

This does not unload the fitted ICA layer matrices.

## Profile fitted components

### `profile_components(...)`

Profile one fitted layer from a stream of raw texts or completed conversations:

```python
profile = lens.profile_components(
    texts_or_conversations,
    layer=5,
    token_scope="all",
    max_tokens=100000,
    top_k_examples=20,
    min_energy=0.05,
    provenance={"dataset": {"repo_id": "owner/dataset", "split": "train"}},
    device="auto",
    progress=True,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `inputs` | iterable of text or conversations, required | Inputs streamed through the analyzed model |
| `layer` | `int`, required | Fitted layer to profile |
| `token_scope` | `str = "all"` | Eligible chat positions; raw text always uses all tokens |
| `max_tokens` | `int \| None = 100000` | Maximum token positions included; `None` consumes the iterable |
| `top_k_examples` | `int = 20` | Highest-energy occurrences retained per component and sign |
| `min_energy` | `float = 0.05` | Minimum per-token component energy required for an example |
| `logit_lens_top_k` | `int = 20` | Top and bottom vocabulary entries retained for each direction |
| `logit_lens_batch_size` | `int = 64` | Writing directions unembedded together; lower this to reduce peak memory |
| `provenance` | `dict \| None = None` | JSON-compatible profiling provenance |
| `context_length` | `int \| None = 1024` | Maximum encoded length of each input |
| `device` | `str \| torch.device \| None = "auto"` | Language-model device |
| `progress` | `bool = False` | Display streaming progress |
| **Returns** | `dict` | Complete per-layer component profile, also attached to the lens |

This is a post-fit operation: it does not alter the fitted center, reading
matrix, or writing matrix. Call `save()` afterward to persist the profile.

### `component_profile(...)`

```python
component = lens.component_profile(layer=5, component=188)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `layer` | `int`, required | Profiled layer |
| `component` | `int`, required | Component ID |
| **Returns** | `dict` | Sign statistics, examples, and logit-lens entries for the component |

Profiles are loaded lazily when reading a local or Hugging Face artifact.

## Transform activation tensors

### `transform(...)`

```python
scores = lens.transform(activations, layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `activations` | floating `torch.Tensor` or `numpy.ndarray`, required | Activation vectors whose final dimension equals `lens.hidden_size` |
| `layer` | `int`, required | Fitted layer transformation to apply |
| **Returns** | same array family as `activations` | Signed ICA scores with the leading dimensions preserved |

`activations` may be a floating-point PyTorch tensor or NumPy array. Its final
dimension must equal `lens.hidden_size`; any leading dimensions are preserved.
The return value has the same array family, device, leading shape, and a final
dimension equal to the number of fitted components.

The operation applies the artifact's preprocessing, fitted center, whitening,
and ICA rotation. The result contains signed ICA scores.

### `energy(...)`

```python
energy = lens.energy(scores)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| **Returns** | same array family as `scores` | Non-negative per-vector component energy shares |

Returns `scores.square() / scores.square().sum(-1, keepdim=True)`. Energy is
non-negative and sums to one for every nonzero score vector. It is a viewing
metric, not an invertible coordinate system.

### `keep_topk(...)`

```python
top_scores = lens.keep_topk(scores, k=10)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| `k` | `int`, required | Number of largest absolute scores to retain per vector |
| **Returns** | same array family as `scores` | Copy with all other components set to zero |

### `ablate_topk(...)`

```python
ablated_scores = lens.ablate_topk(scores, k=10)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed component scores |
| `k` | `int`, required | Number of largest absolute scores to remove per vector |
| **Returns** | same array family as `scores` | Copy with the selected components set to zero |

Both methods rank components independently at every token position and do not
modify their inputs in place.

### `inverse_transform(...)`

```python
reconstructed = lens.inverse_transform(top_scores, layer=6)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `scores` | floating `torch.Tensor` or `numpy.ndarray`, required | Signed scores whose final dimension equals the fitted component count |
| `layer` | `int`, required | Fitted writing matrix and center to apply |
| **Returns** | same array family as `scores` | Reconstructed preprocessed activations |

Maps signed scores back through the writing matrix and fitted center. The
result reconstructs the preprocessed activation, so only signed scores—not
energy shares—are valid input.

### `restore_norm(...)`

```python
hidden_states = lens.restore_norm(
    reconstructed,
    reference=activations,
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `values` | floating `torch.Tensor` or `numpy.ndarray`, required | Reconstructed or modified activation directions |
| `reference` | same array type and shape, required | Activations whose per-vector norms should be restored |
| **Returns** | same array family as `values` | `values` rescaled to the reference norms |

Rescales each reconstructed vector to the norm of its corresponding reference
activation. `values` and `reference` must have identical shapes and must both
be PyTorch tensors or both be NumPy arrays.

## Fit activation tensors

### `fit(...)`

```python
lens.fit(
    activations,
    layer=6,
    n_components=None,
    max_iter=20,
    random_state=0,
    progress=True,
    device="cuda",
    batch_size=8192,
    objective_every=1,
    provenance={
        "dataset": {"repo_id": "owner/dataset", "split": "train"},
        "fitting_tokens": int(activations.shape[0]),
    },
)
```

`fit()` fits or replaces one layer and returns the same lens object.

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `activations` | floating `torch.Tensor` or `numpy.ndarray`, required | Activations; leading dimensions are flattened into samples |
| `layer` | `int`, required | Label for the source layer; `fit()` does not capture activations |
| `n_components` | `int \| None = None` | Component count; defaults to the activation width |
| `algorithm` | `str = "parallel"` | FastICA update algorithm; currently only `"parallel"` |
| `fun` | `str = "logcosh"` | Contrast function; currently only `"logcosh"` |
| `max_iter` | `int = 200` | Fixed FastICA iteration count; no tolerance-based early stopping |
| `random_state` | `int \| None = 0` | FastICA initialization seed |
| `progress` | `bool = False` | Show preprocessing and optimization progress |
| `device` | `str \| torch.device \| None = None` | Fitting device; defaults to the activation tensor's device |
| `batch_size` | `int = 8192` | Activation rows processed on the fitting device at once |
| `objective_every` | `int = 1` | Record objective percentiles every N iterations |
| `provenance` | `dict \| None = None` | JSON-compatible fitting metadata stored verbatim in the artifact |
| **Returns** | `ICALens` | The same lens, with this layer fitted or replaced |

A full-width fit requires at least `hidden_size + 1` samples because centering
removes one degree of rank. Calling `fit()` again with another layer adds it to
the same artifact. See [Fit and publish](fit-and-publish.md) for end-to-end
dataset commands and a complete Python example.

## Save and publish

### `save(...)`

```python
path = lens.save("icalens-output/my-icalens")
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `path` | `str \| Path`, required | Destination artifact directory |
| **Returns** | `Path` | Resolved destination directory |

Writes the manifest, fitted layer tensors, and generated model card. It returns
the resolved output `Path`. Local output paths are not serialized, but values
placed in `provenance` are stored verbatim.

### `push_to_hub(...)`

```python
url = lens.push_to_hub(
    "owner/my-icalens",
    private=False,
    revision="main",
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `repo_id` | `str`, required | Destination in `owner/name` form |
| `private` | `bool \| None = None` | Requested visibility when creating the repository |
| `token` | `str \| bool \| None = None` | Hugging Face authentication setting |
| `revision` | `str = "main"` | Destination branch or revision |
| `commit_message` | `str = "Upload ICA Lens artifacts"` | Hub commit message |
| **Returns** | `str` | URL of the created Hub commit |

Creates or updates a Hugging Face **Model** repository and returns the commit
URL. Authentication follows `huggingface_hub`; use `hf auth login`, `HF_TOKEN`,
or pass `token=` explicitly.

## Results

### `CaptureResult`

`capture()` returns token-aligned model data:

| Field | Contents |
| --- | --- |
| `tokens` | Tokenizer-native token strings |
| `token_texts` | Readable token text used by reports |
| `token_labels` | Compact token-card labels |
| `token_tooltips` | Expanded text shown on hover |
| `token_groups` | Conversation group label for each selected token; empty for raw text |
| `token_ids` | Selected token IDs |
| `positions` | Positions in the complete encoded model input |
| `activations` | Captured activation vectors |

### `AnalysisResult`

`AnalysisResult` contains every `CaptureResult` field plus:

| Field | Contents |
| --- | --- |
| `scores` | Signed ICA component scores |
| `energy` | Per-token component energy shares |
| `model` | Model repository and exact revision |
| `layer` | Analyzed transformer-block index |
| `input_text` | Readable input representation |
| `token_scope` | Positions selected from the encoded input |
| `messages` | Original conversation messages, or an empty tuple for text |

In Jupyter or Colab, leave `result` as the final expression to render it:

```python
result
```

### `AnalysisResult.display(...)`

Configure the inline view explicitly with:

```python
result.display(metric="energy", top_k=5, height=720)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `metric` | `"score" \| "energy" = "score"` | Values shown in component bars |
| `top_k` | `int = 3` | Components displayed per token card |
| `title` | `str = "ICA Lens Explorer"` | Embedded report title |
| `height` | `int = 720` | Initial iframe height in pixels |
| **Returns** | `None` | Displays the interactive result through IPython |

### `AnalysisResult.to_html(...)`

Write a self-contained HTML explorer outside a notebook with:

```python
output = result.to_html(
    "analysis.html",
    metric="score",
    top_k=3,
    title="ICA Lens Explorer",
)
```

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `output_file` | `str \| Path`, required | Destination HTML file |
| `metric` | `"score" \| "energy" = "score"` | Values shown in component bars |
| `top_k` | `int = 3` | Components displayed per token card |
| `title` | `str = "ICA Lens Explorer"` | Report title |
| **Returns** | `Path` | Resolved path of the self-contained report |

## Inspect a lens

```python
print(lens.model_id)
print(lens.model_revision)
print(lens.model_type)
print(lens.activation_site)
print(lens.hidden_size)
print(lens.available_layers)
print(lens.metadata)
```

`metadata` returns a detached dictionary representing the portable manifest.

## Exceptions

```python
from icalens import ArtifactError, ICALensError, NotFittedError
```

- `ICALensError` is the base package exception.
- `ArtifactError` reports missing, malformed, or incompatible artifacts.
- `NotFittedError` reports a requested layer that has not been fitted or
  loaded.
