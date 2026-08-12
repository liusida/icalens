# ICA Lens

ICA Lens fits, shares, and applies Independent Component Analysis bases for
language-model activations. It can capture activations directly from text or
operate on activation tensors supplied by the caller.

```bash
uv add icalens
```

Load a published lens:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)

print(result.tokens)
print(result.scores)
result.to_html("analysis.html")
```

Fit and publish your own:

```python
from icalens import ICALens

lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
)
lens.fit(activations, layer=6, random_state=0)
lens.save("./my-icalens")
lens.push_to_hub("username/icalens-gpt2-small")
```

For the standalone publishing demo, create a project-root `.env` file containing
a Hugging Face token with write permission:

```dotenv
HF_TOKEN=hf_...
```

The `.env` file is ignored by Git. Publish a saved lens with:

```bash
uv run python demo/publish.py \
  --lens ./my-icalens \
  username/icalens-model-name
```

Instruction-tuned checkpoints use the same activation-level API and are
identified explicitly in their portable metadata:

```python
lens = ICALens(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_type="instruct",
)
```

`model_type` describes the checkpoint and accepts `"base"` or `"instruct"`.
The standard `icalens` installation can capture and analyze text or completed
chat conversations directly. `result = lens.analyze(text, layer=6)` returns aligned
tokens, activations, signed scores, and per-token component energy shares.
`result.to_html("analysis.html")` writes a self-contained interactive explorer;
pass `metric="energy"` to visualize component energy shares instead of scores.

Inputs may be NumPy arrays or PyTorch tensors. Leading dimensions are treated
as sample dimensions and the final dimension must be the model hidden size.
Fitting uses ICA Lens's built-in PyTorch FastICA implementation and can run on
the input tensor's device. NumPy inputs are fitted on CPU. ICA Lens does not
depend on scikit-learn or SciPy.

See the [documentation](https://icalens.readthedocs.io/) for the complete text,
conversation, fitting, publishing, and HTML-export workflows.

For the 1,000-token GPT-2/Pile-10k fitting demo, run:

```bash
uv sync
uv run python demo/fit.py
```

For the corresponding instruct-model demo using all formatted UltraChat
conversation tokens, including template markers, run:

```bash
uv run python demo/fit_chat.py --layers 12
```

Then inspect assistant-token component scores with:

```bash
uv run python demo/apply_chat.py
```

Both `apply.py` and `apply_chat.py` also write standalone interactive HTML
explorers under `demo/output/`; pass `--output-file` to choose another path.

## Installed-package smoke test

After installing a wheel or PyPI release in a clean project, run the bundled
end-to-end check:

```bash
uv run icalens-smoke-test
```

By default, the suite checks both public input paths: raw text through the
published GPT-2 lens and a formatted conversation through the published
Qwen3.5-2B instruct lens. Each case lazily downloads one ICA layer, verifies
finite scores and normalized energy, checks reconstruction shape, and writes
`icalens-smoke-text.html` or `icalens-smoke-chat.html`.

Run only one path when iterating locally:

```bash
uv run icalens-smoke-test text
uv run icalens-smoke-test chat
```

Use `--text-lens`, `--text-layer`, `--text-input`, `--chat-lens`,
`--chat-layer`, `--chat-input`, `--chat-response`, `--device`, and
`--output-dir` to override the defaults.
