# ICA Lens

ICA Lens fits, shares, and applies Independent Component Analysis bases for
language-model activations. Version 0.1 operates on activations supplied by the
caller; it does not load language models or capture activations.

```bash
uv add icalens
```

Load a published lens:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("liusida/icalens-gpt2-small")
scores = lens.transform(activations, layer=6)
reconstructed = lens.inverse_transform(scores, layer=6)
```

Fit and publish your own:

```python
from icalens import ICALens

lens = ICALens(
    base_model="openai-community/gpt2",
    base_model_revision="FULL_COMMIT_HASH",
    activation_site="resid_post",
)
lens.fit(activations, layer=6, random_state=0)
lens.save("./my-icalens")
lens.push_to_hub("username/icalens-gpt2-small")
```

Inputs may be NumPy arrays or PyTorch tensors. Leading dimensions are treated
as sample dimensions and the final dimension must be the model hidden size.
Fitting uses ICA Lens's built-in PyTorch FastICA implementation and can run on
the input tensor's device. NumPy inputs are fitted on CPU. ICA Lens does not
depend on scikit-learn or SciPy.

See [`docs/api.md`](docs/api.md) and
[`docs/artifact-format.md`](docs/artifact-format.md) for the initial API and
portable artifact format.

For the 1,000-token GPT-2/Pile-10k fitting demo, run:

```bash
uv sync
uv run python demo/fit.py
```
