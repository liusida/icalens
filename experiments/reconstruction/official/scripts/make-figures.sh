#!/usr/bin/env bash
set -euo pipefail

ROOT="experiments/reconstruction/pilot-runs/upgrade-full-run"

uv run icalens experiment figure reconstruction \
  "$ROOT/results/gpt2" \
  "$ROOT/results/gemma2" \
  "$ROOT/results/qwen35-9b" \
  --panel-titles "GPT-2 Small,Gemma 2 2B,Qwen 3.5 9B Base" \
  --output "$ROOT/figures" \
  --format png,pdf \
  --force
