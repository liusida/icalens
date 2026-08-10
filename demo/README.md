# ICA Lens fitting demo

Install the development dependencies and fit a lens from 1,000 Pile-10k tokens:

```bash
uv sync
uv run python demo/fit.py
```

By default, the demo:

1. Streams text from `NeelNanda/pile-10k`.
2. Builds a 1,000-token candidate pool from independently tokenized documents.
3. Uses all 1,000 positions for fitting by default.
4. Preserves each sampled token's original left context during activation capture.
5. Loads `openai-community/gpt2` on CUDA through `gb10-load-llm`.
6. Captures transformer block 6's direct output (`resid_post`) with a forward
   hook, before any model-level final normalization.
7. Fits a full 768-component ICA Lens.
8. Saves it under `demo/output/icalens-gpt2-small-1k`.

By default, `--candidate-tokens` equals `--token-budget`. Set it explicitly to
sample the fitting tokens from a larger pool:

```bash
uv run python demo/fit.py \
  --candidate-tokens 10000 \
  --token-budget 1000 \
  --max-iter 1000 \
  --seed 0
```

`--max-iter` controls the fixed number of FastICA fixed-point iterations.
The fitting demo displays a tqdm bar with the current convergence limit and
mean contrast objective (`obj`). ICA Lens deliberately runs exactly
`--max-iter` iterations: the classical FastICA tolerance criterion is treated
as diagnostic-only because it is not an appropriate stopping rule in the LLM
activation regime. The displayed objective uses the selected FastICA contrast:
`log(cosh(x))`, `-exp(-x²/2)`, or `x⁴/4`.

The demo also displays token-rate progress bars while building the Pile-10k
candidate pool and capturing the sampled GPT-2 activations.

To test whether a fit stays within an ordinary 16 GiB GPU budget, cap the
PyTorch CUDA allocator:

```bash
uv run python demo/fit.py \
  --layers 6 \
  --token-budget 1000000 \
  --fit-batch-size 8192 \
  --max-vram-gb 16
```

The script reports peak PyTorch CUDA memory at the end. This cap covers
allocations managed by PyTorch, but not CUDA context memory or allocations made
directly by non-PyTorch libraries, so it is a close OOM test rather than a full
hardware emulator.

Selected activations are retained in CPU memory. Whitening statistics,
FastICA updates, and final source scaling are computed in repeated CUDA batches,
so GPU memory scales with `--fit-batch-size` rather than the total token count.
Set `--fit-batch-size 0` to disable batching and process all selected activation
rows at once. This can be faster when sufficient GPU memory is available, but
memory then scales with the full token budget. Negative values are invalid.

Both demos intentionally fit a full hidden-size set of ICA components. They
stop with a clear error when `--token-budget` is smaller than the model hidden
size plus one rather than silently producing a reduced-dimensional lens. The
extra sample is required because centering limits rank to at most
`n_samples - 1`.

Fit several layers with a comma-separated list:

```bash
uv run python demo/fit.py --layers 0,6,11
```

Use `--layers all` to fit all 12 GPT-2 transformer blocks. The default single
layer is intended as a quick end-to-end check; a full published artifact should
use a larger, explicitly sampled token corpus.

The demo requires network access for Hugging Face downloads and a CUDA device.

## Fit an instruct-model lens from conversations

Fit a Qwen2.5-0.5B-Instruct lens on assistant-content tokens from streamed
UltraChat 200k conversations:

```bash
uv run python demo/fit_chat.py --layers 12
```

The script uses the tokenizer's chat template and offset mapping to distinguish
message content from role markers and other template control tokens. The
default `--token-scope assistant` therefore fits only assistant-content token
activations while retaining the complete preceding conversation as context.
Other supported scopes are `user`, `content` (all message content), and `all`
(including template tokens).

As in the plain-text demo, `--candidate-tokens` defaults to `--token-budget`.
For a larger deterministic candidate pool, run:

```bash
uv run python demo/fit_chat.py \
  --candidate-tokens 100000 \
  --token-budget 10000 \
  --max-iter 100
```

The model, dataset, split, message-field, context length, output path, fitting
batch size, and CUDA allocator cap are configurable. For example:

```bash
uv run python demo/fit_chat.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --messages-field messages \
  --token-scope assistant \
  --context-length 1024 \
  --layers 12 \
  --max-vram-gb 16
```

Ask the model to generate a response and apply the resulting instruct lens to
the generated assistant tokens:

```bash
uv run python demo/apply_chat.py
```

The script first generates a response, then performs a full forward pass over
the completed conversation so each generated token has an activation aligned
with the same chat formatting used during fitting. The default output shows
component scores only for generated assistant-content tokens. Pass `--user`
and optionally `--system` to supply a different prompt. `--token-scope` accepts
the same `assistant`, `user`, `content`, and `all` choices as the fitting demo:

```bash
uv run python demo/apply_chat.py \
  --user "What is an eigenvector?" \
  --max-new-tokens 128 \
  --token-scope assistant \
  --layer 12
```

The command also writes a self-contained v5-style explorer to
`demo/output/apply_chat.html` by default. Use `--output-file PATH` to choose a
different location. The report works directly from disk and does not require
the v5 server.

## Apply the saved lens

After fitting layer 6, apply it to fresh text:

```bash
uv run python demo/apply.py
```

The script loads the local artifact, captures GPT-2 activations from the exact
model revision recorded in its manifest, and prints the largest signed ICA
component scores at each token.

Use custom text or another saved artifact with:

```bash
uv run python demo/apply.py \
  --lens demo/output/icalens-gpt2-small-1k \
  --layer 6 \
  --text "The boat reached the bank before sunset."
```

The command writes `demo/output/apply.html` by default. The standalone HTML
includes responsive token cards, signed component bars, component highlighting,
card-width control, and an opacity cutoff. Override the path with
`--output-file PATH`.

## Publish and verify a chat lens

The fitting demo records exact dataset and sampling provenance. Upload through
the public cloud API with an explicit opt-in:

```bash
uv run python demo/fit_chat.py \
  --layers 12 \
  --push-to-hub username/icalens-qwen2.5-0.5b-instruct
```

Add `--private` for a private Hugging Face Model repository. The demo reloads
the uploaded lens and verifies its scores against the local artifact.

Any saved lens can also be published independently:

```bash
uv run python demo/publish.py sida/icalens-gpt2-small \
  --lens demo/output/icalens-gpt2-small
```

Authentication uses Hugging Face Hub's standard credentials. Run
`hf auth login` once, or provide the `HF_TOKEN` environment variable. The
script never accepts or prints the token itself; it only reports the account
name returned by Hugging Face.
