# ICA Lens

ICA Lens fits, shares, and applies Independent Component Analysis bases for
language-model activations. The core API operates on activations supplied by
the caller; it does not load language models or capture activations.

```bash
uv add icalens
```

Load a published lens:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("liusida/icalens-gpt2-small")
scores = lens.transform(activations, layer=6)
reconstructed = lens.inverse_transform(scores, layer=6)
energy = lens.energy(scores)  # per-token component fractions summing to 1
```

Fit and publish your own:

```python
from icalens import ICALens

lens = ICALens(
    model_id="openai-community/gpt2",
    model_revision="FULL_COMMIT_HASH",
    model_type="base",
    activation_site="resid_post",
)
lens.fit(activations, layer=6, random_state=0)
lens.save("./my-icalens")
lens.push_to_hub("username/icalens-gpt2-small")
```

Instruction-tuned checkpoints use the same activation-level API and are
identified explicitly in their portable metadata:

```python
lens = ICALens(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_revision="FULL_COMMIT_HASH",
    model_type="instruct",
)
```

`model_type` describes the checkpoint and accepts `"base"` or `"instruct"`.
Install `icalens[analyze]` to capture and analyze text or completed chat
conversations directly. `result = lens.analyze(text, layer=6)` returns aligned
tokens, activations, signed scores, and per-token component energy shares.

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

For the corresponding instruct-model demo using assistant tokens from
UltraChat conversations, run:

```bash
uv run python demo/fit_chat.py --layers 12
```

Then inspect assistant-token component scores with:

```bash
uv run python demo/apply_chat.py
```

Both `apply.py` and `apply_chat.py` also write standalone interactive HTML
explorers under `demo/output/`; pass `--output-file` to choose another path.
