#!/usr/bin/env bash
set -euo pipefail

ROOT="experiments/reconstruction/official"

uv run icalens experiment figure reconstruction \
  "$ROOT/results/gpt2" \
  "$ROOT/results/gemma2" \
  "$ROOT/results/qwen35-9b" \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base" \
  --output "$ROOT/figures" \
  --format png,pdf \
  --force

for spec in \
  "gpt2|GPT-2 Small" \
  "gemma2|Gemma 2 2B" \
  "qwen35-9b|Qwen 3.5 9B Base"
do
  model="${spec%%|*}"
  title="${spec#*|}"
  uv run icalens experiment figure reconstruction \
    "$ROOT/results/$model" \
    --panel-titles "$title" \
    --output "$ROOT/figures/$model-preview" \
    --format png \
    --force
done
