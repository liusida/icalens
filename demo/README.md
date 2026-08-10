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

For parallel FastICA, every saved layer also records an objective curve in
`icalens.json` under `layers.<layer>.fitting.objective_history`. At each
recorded iteration, the contrast is averaged over fitting tokens for each
component, then summarized across components at the minimum, 10th, 20th, ...,
90th percentile, and maximum. Its `iterations`, `percentiles`, and `values`
arrays can be plotted directly as percentile curves or nested colored bands.
Use `--objective-every N` to record every Nth iteration; the final iteration is
always included. The default is `1`.

Plot the first four available layers from a local, progressively written lens:

```bash
uv run python demo/plot_objective.py \
  demo/output/icalens-qwen3.5-2b-ultrachat-10m
```

Use `--layers 0,1,2`, `--first 6`, or `--output path/to/curves.png` to customize
the selection and output. Nested colored bands show min–max, p10–p90, ...,
p40–p60, with the median drawn as a line.

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

Selected activations are retained in CPU memory in the model's activation
dtype. Whitening statistics and FastICA updates are computed in repeated CUDA batches,
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

Limit CPU activation memory by capturing and fitting only a few layers per
model pass:

```bash
uv run python demo/fit.py \
  --layers all \
  --token-budget all \
  --capture-layers-at-once 2
```

This captures layers 0–1, fits them, releases their activations, and then
repeats for layers 2–3. Smaller groups use less memory but require more complete
passes over the tokenized dataset. Each capture pass stops immediately after
its highest requested transformer block, so an early-layer group does not run
the unused remainder of the model. The default `0` captures all requested
layers in one pass.

After each layer finishes fitting, the demos atomically checkpoint the growing
lens to `--output`, including that layer's objective history. If a later layer
fails or the run is interrupted, every previously completed layer remains
loadable from the output directory.

The demo requires network access for Hugging Face downloads and a CUDA device.

Use every token available under the demo's per-document context limit without
knowing the count in advance:

```bash
uv run python demo/fit.py --layers 6 --token-budget all
```

The resolved usable-token count is printed before model loading. Capturing many
layers simultaneously retains one BF16 activation matrix per layer for the
default model, so use `--capture-layers-at-once` when the full set would exceed
system memory. Both fitting demos report peak process RSS and peak PyTorch CUDA
reserved memory at the end.

## Fit an instruct-model lens from conversations

Fit a Qwen2.5-0.5B-Instruct lens on all formatted tokens from streamed
UltraChat 200k conversations:

```bash
uv run python demo/fit_chat.py --layers 12
```

Qwen3.5 multimodal checkpoints can be fitted as text-only language models. The
loader selects the language backbone, while chat templating and assistant-token
selection use the same interface:

```bash
uv run python demo/fit_chat.py \
  --model Qwen/Qwen3.5-2B \
  --layers 12 \
  --token-budget 100000 \
  --max-iter 20 \
  --output demo/output/icalens-qwen3.5-2b-ultrachat-100k
```

Qwen3.5-2B has 24 language layers indexed from 0 through 23 and hidden size
2048. Use at least 2049 fitting tokens for a full-component lens.

The script uses the tokenizer's chat template and offset mapping to distinguish
message content from role markers and other template control tokens. The
default `--token-scope all` fits every formatted position, including template
tokens. Other supported scopes are `assistant`, `user`, and `content` (all
message content but no template markers).

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
the complete formatted conversation:

```bash
uv run python demo/apply_chat.py
```

The script first generates a response, then performs a full forward pass over
the completed conversation so each formatted token has an activation aligned
with the same chat formatting used during fitting. The default output shows
component scores for all formatted tokens. Pass `--user` and optionally
`--system` to supply a different prompt. `--token-scope` accepts the same
`assistant`, `user`, `content`, and `all` choices as the fitting demo:

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

By default, the publishing demo reads `HF_TOKEN` from the project-root `.env`
file. That file is ignored by Git. If it is absent or does not define
`HF_TOKEN`, Hugging Face Hub's standard environment variable or saved `hf auth
login` credential is used instead. Pass `--env-file PATH` to select another
dotenv file. The script never accepts or prints the token itself; it only
reports the account name returned by Hugging Face.
