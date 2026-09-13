#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment figure sparse-probing \
  experiments/sparse-probing/official/results/gpt2 \
  experiments/sparse-probing/official/results/gemma2 \
  - \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base (pending)" \
  --output experiments/sparse-probing/official/figures \
  --format png,pdf \
  --force
