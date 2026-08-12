# API

## `ICALens`

```python
ICALens(
    *,
    model_id: str,
    model_revision: str | None = None,
    model_type: Literal["base", "instruct"] = "base",
    activation_site: str = "resid_post",
    row_normalize: bool = True,
)
```

`model_revision` is optional. When ICA Lens loads a model for capture or
analysis, it resolves and records the current exact revision automatically.

### Load

```python
lens = ICALens.from_pretrained("owner/repository")
```

The source may be a Hugging Face Model repository or a local artifact
directory. Layer tensors are downloaded lazily when used.

### Capture and analyze

```python
captured = lens.capture(inputs, layer=6)
result = lens.analyze(inputs, layer=6)
```

`inputs` may be raw text or a completed chat conversation. `capture()` returns
aligned tokens, token IDs, positions, and activations. `analyze()` additionally
returns signed scores and energy shares. Both use `device="auto"` by default,
selecting CUDA when available and CPU otherwise. Pass `device="cuda"` or
`device="cpu"` to choose explicitly.

### Transform

```python
scores = lens.transform(activations, layer=6)
reconstructed = lens.inverse_transform(scores, layer=6)
energy = lens.energy(scores)
```

### Fit and save

```python
lens.fit(activations, layer=6)
lens.save("./my-icalens")
lens.push_to_hub("owner/my-icalens")
```

### Inspect

```python
print(lens.model_id)
print(lens.model_revision)
print(lens.model_type)
print(lens.available_layers)
print(lens.metadata)
```

## `AnalysisResult`

Important fields are:

- `tokens`
- `token_texts`
- `token_ids`
- `positions`
- `activations`
- `scores`
- `energy`

Export an interactive report with:

```python
result.to_html(
    "analysis.html",
    metric="score",
    top_k=5,
)
```
