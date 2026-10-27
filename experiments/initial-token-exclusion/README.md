# Initial-token exclusion

This experiment compares the released GPT-2 Small Pile-10k ICA Lens, whose raw documents use
the registered prepended `<|endoftext|>` boundary context, with a fit that uses native unframed
documents and excludes the first two document-token positions from ICA sampling.

The new fit keeps those two positions in the model forward pass as left context. It excludes only
their activation rows from the candidate pool.

## Fit

Run from the repository root:

```bash
uv run icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --context-length 1024 \
  --document-framing none \
  --skip-initial-tokens 2 \
  --candidate-tokens all \
  --token-budget 1000000 \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 32768 \
  --capture-layers-at-once 1 \
  --seed 0 \
  --output experiments/initial-token-exclusion/runs/gpt2-no-framing-skip-2
```

This matches the released GPT-2 Lens settings except for the intended treatment of initial
positions. The output provenance records both `document_framing` and `skip_initial_tokens`.

## Comparison target

Compare against:

```text
local-icalens-models/official/icalens-gpt2-small-pile10k
```

At minimum, compare matched reading directions up to permutation and sign, component strength and
kurtosis, and whether components or profile extremes are dominated by the first positions. Treat
interpretability or fitting-quality claims as pending until those comparisons are complete.

## Chat-model fits

Chat fits keep each model's native rendered chat template, but exclude rendered positions 0 and 1
from the fitting candidate pool. Use layers 0, 12, and 25 for Gemma 2 2B IT:

```bash
uv run icalens fit chat \
  --model google/gemma-2-2b-it \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --messages-field messages \
  --token-scope all \
  --skip-initial-tokens 2 \
  --layers 0,12,25 \
  --context-length 1024 \
  --candidate-tokens 1000000 \
  --token-budget 1000000 \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 32768 \
  --capture-layers-at-once 1 \
  --seed 0 \
  --output experiments/initial-token-exclusion/runs/gemma-2-2b-it-skip-2
```

Use layers 0, 11, and 23 for Qwen 3.5 2B:

```bash
uv run icalens fit chat \
  --model Qwen/Qwen3.5-2B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --messages-field messages \
  --token-scope all \
  --skip-initial-tokens 2 \
  --layers 0,11,23 \
  --context-length 1024 \
  --candidate-tokens 10000000 \
  --token-budget 1000000 \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 8192 \
  --capture-layers-at-once 1 \
  --seed 0 \
  --output experiments/initial-token-exclusion/runs/qwen3.5-2b-chat-skip-2
```
