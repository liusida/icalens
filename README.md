# ICA Lens

ICA Lens fits, shares, and applies Independent Component Analysis bases for
language-model activations. It can capture activations directly from text or
operate on activation tensors supplied by the caller.

By [Sida Liu](https://liusida.com/) and
[Feijiang Han](https://feijianghan.com/).

```bash
uv add icalens
```

Load a published lens:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)

result
```

Fit and publish your own:

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --layers 6 \
  --token-budget 1000 \
  --output ./my-icalens
```

To publish from the command line, authenticate with `hf auth login`, set
`HF_TOKEN`, or create a `.env` file in the current directory containing a
Hugging Face token with write permission:

```dotenv
HF_TOKEN=hf_...
```

Publish a saved lens with:

```bash
icalens publish \
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

Run a small GPT-2/Pile-10k fitting check with the installed text command:

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers 6 \
  --token-budget 1000 \
  --max-iter 20
```

Fit an instruction-tuned model from UltraChat conversations with:

```bash
icalens fit chat \
  --model Qwen/Qwen3.5-2B \
  --layers 12 \
  --token-budget 100000
```

The source repository retains thin `demo/` wrappers for contributors, but they
are not required when using the installed package.

## Installed-package smoke test

After installing a wheel or PyPI release in a clean project, run the bundled
end-to-end check:

```bash
uv run icalens smoke-test
```

By default, the suite checks both public input paths: raw text through the
published GPT-2 lens and a formatted conversation through the published
Qwen3.5-2B instruct lens. Each case lazily downloads one ICA layer, verifies
finite scores and normalized energy, checks reconstruction shape, and writes
`icalens-smoke-text.html` or `icalens-smoke-chat.html`.

Run only one path when iterating locally:

```bash
uv run icalens smoke-test text
uv run icalens smoke-test chat
```

Use `--text-lens`, `--text-layer`, `--text-input`, `--chat-lens`,
`--chat-layer`, `--chat-input`, `--chat-response`, `--device`, and
`--output-dir` to override the defaults.
