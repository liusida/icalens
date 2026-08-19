#!/usr/bin/env bash
set -euo pipefail

# Refit the six published ICA Lens models without ICA Lens row normalization.
# Models are fitted sequentially. Completed models remain available if a later
# command fails; rerun individual commands as needed rather than this whole file.

OUTPUT_ROOT="local-icalens-models/current"
ACTIVATION_ROOT="${HOME}/data/icalens-activations"
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$ACTIVATION_ROOT"

# 1. GPT-2 Small and 2. Gemma 2 2B have already completed.

# 3. Qwen 3.5 9B Base (base, Pile-10k)
echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [3/6] Capturing Qwen 3.5 9B Base (base, Pile-10k)"
echo
uv run icalens capture text \
  --model Qwen/Qwen3.5-9B-Base \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --document-framing auto \
  --layers all \
  --candidate-tokens all \
  --token-budget 1000000 \
  --capture-layers-at-once all \
  --output "$ACTIVATION_ROOT/qwen3.5-9b-base-pile10k-1m"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [3/6] Fitting Qwen 3.5 9B Base from captured activations"
echo
uv run icalens fit activations \
  --input "$ACTIVATION_ROOT/qwen3.5-9b-base-pile10k-1m" \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 16384 \
  --output "$OUTPUT_ROOT/icalens-qwen3.5-9b-base-pile10k"

# 4. Gemma 2 2B IT (instruct, UltraChat)
echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [4/6] Capturing Gemma 2 2B IT (instruct, UltraChat)"
echo
uv run icalens capture chat \
  --model google/gemma-2-2b-it \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --messages-field messages \
  --token-scope all \
  --layers all \
  --candidate-tokens 1000000 \
  --token-budget 1000000 \
  --capture-layers-at-once all \
  --output "$ACTIVATION_ROOT/gemma-2-2b-it-ultrachat-1m"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [4/6] Fitting Gemma 2 2B IT from captured activations"
echo
uv run icalens fit activations \
  --input "$ACTIVATION_ROOT/gemma-2-2b-it-ultrachat-1m" \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 32768 \
  --output "$OUTPUT_ROOT/icalens-gemma-2-2b-it-ultrachat-1m"

# 5. Qwen 3.5 2B Base (base, Pile-10k)
echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [5/6] Capturing Qwen 3.5 2B Base (base, Pile-10k)"
echo
uv run icalens capture text \
  --model Qwen/Qwen3.5-2B-Base \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --document-framing auto \
  --layers all \
  --candidate-tokens all \
  --token-budget 1000000 \
  --capture-layers-at-once all \
  --output "$ACTIVATION_ROOT/qwen3.5-2b-base-pile10k-1m"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [5/6] Fitting Qwen 3.5 2B Base from captured activations"
echo
uv run icalens fit activations \
  --input "$ACTIVATION_ROOT/qwen3.5-2b-base-pile10k-1m" \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 32768 \
  --output "$OUTPUT_ROOT/icalens-qwen3.5-2b-base-pile10k"

# 6. Qwen 3.5 2B (instruct, UltraChat)
echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [6/6] Capturing Qwen 3.5 2B (instruct, UltraChat)"
echo
uv run icalens capture chat \
  --model Qwen/Qwen3.5-2B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --messages-field messages \
  --token-scope all \
  --layers all \
  --candidate-tokens 10000000 \
  --token-budget 1000000 \
  --capture-layers-at-once all \
  --output "$ACTIVATION_ROOT/qwen3.5-2b-ultrachat-1m"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [6/6] Fitting Qwen 3.5 2B from captured activations"
echo
uv run icalens fit activations \
  --input "$ACTIVATION_ROOT/qwen3.5-2b-ultrachat-1m" \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 8192 \
  --output "$OUTPUT_ROOT/icalens-qwen3.5-2b-ultrachat-1m"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Finished all 6 raw ICA Lens refits."
