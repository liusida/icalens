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
6. Captures `outputs.hidden_states[7]`, recorded as ICA Lens layer 6 after
   excluding the initial embedding state.
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

Fit several layers with a comma-separated list:

```bash
uv run python demo/fit.py --layers 0,6,11
```

Use `--layers all` to fit all 12 GPT-2 transformer blocks. The default single
layer is intended as a quick end-to-end check; a full published artifact should
use a larger, explicitly sampled token corpus.

The demo requires network access for Hugging Face downloads and a CUDA device.

## Apply the saved lens

After fitting layer 6, apply it to fresh text:

```bash
uv run python demo/apply.py
```

The script loads the local artifact, captures GPT-2 activations from the exact
base-model revision recorded in its manifest, and prints the largest signed ICA
component scores at each token.

Use custom text or another saved artifact with:

```bash
uv run python demo/apply.py \
  --lens demo/output/icalens-gpt2-small-1k \
  --layer 6 \
  --text "The boat reached the bank before sunset."
```
