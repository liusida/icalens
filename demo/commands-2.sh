#!/usr/bin/env bash
set -euo pipefail

# Refit the three base-model ICA lenses with model-aware document framing.
# Each fit samples 1M rows from all usable Pile-10k tokens. New output paths
# preserve the existing artifacts until these corrected lenses are validated.
#
# Gemma prerequisite: accept the license for google/gemma-2-2b on Hugging Face.

echo "[1/6] Fitting GPT-2 Small"
uv run icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --capture-layers-at-once 5 \
  --candidate-tokens all \
  --token-budget 1000000 \
  --fit-batch-size 32768 \
  --max-iter 50 \
  --refresh-model-registry \
  --output three-icalens-fit/icalens-gpt2-small-pile10k-1m

echo "[2/6] Profiling GPT-2 Small"
uv run icalens profile \
  --lens three-icalens-fit/icalens-gpt2-small-pile10k-1m \
  --layers all \
  --dataset NeelNanda/pile-10k \
  --split train \
  --input-type text \
  --text-field text \
  --max-tokens 1000000 \
  --top-k-examples 20 \
  --min-energy 0.05

echo "[3/6] Fitting Gemma 2 2B Base"
uv run icalens fit text \
  --model google/gemma-2-2b \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --capture-layers-at-once 5 \
  --candidate-tokens all \
  --token-budget 1000000 \
  --fit-batch-size 32768 \
  --max-iter 50 \
  --refresh-model-registry \
  --output three-icalens-fit/icalens-gemma-2-2b-pile10k-1m

echo "[4/6] Profiling Gemma 2 2B Base"
uv run icalens profile \
  --lens three-icalens-fit/icalens-gemma-2-2b-pile10k-1m \
  --layers all \
  --dataset NeelNanda/pile-10k \
  --split train \
  --input-type text \
  --text-field text \
  --max-tokens 1000000 \
  --top-k-examples 20 \
  --min-energy 0.05

echo "[5/6] Fitting Qwen3.5 2B Base"
uv run icalens fit text \
  --model Qwen/Qwen3.5-2B-Base \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --capture-layers-at-once 5 \
  --candidate-tokens all \
  --token-budget 1000000 \
  --fit-batch-size 32768 \
  --max-iter 50 \
  --refresh-model-registry \
  --output three-icalens-fit/icalens-qwen3.5-2b-base-pile10k-1m

echo "[6/6] Profiling Qwen3.5 2B Base"
uv run icalens profile \
  --lens three-icalens-fit/icalens-qwen3.5-2b-base-pile10k-1m \
  --layers all \
  --dataset NeelNanda/pile-10k \
  --split train \
  --input-type text \
  --text-field text \
  --max-tokens 1000000 \
  --top-k-examples 20 \
  --min-energy 0.05

echo "Finished fitting and profiling all three corrected base-model lenses."
