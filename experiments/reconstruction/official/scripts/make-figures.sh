#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment figure reconstruction \
  experiments/reconstruction/official/results/gpt2 \
  experiments/reconstruction/official/results/gemma2 \
  experiments/reconstruction/official/results/qwen35-9b \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base" \
  --output experiments/reconstruction/official/figures \
  --format png,pdf \
  --force
