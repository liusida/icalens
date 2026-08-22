#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment reconstruction \
  --lens sida/icalens-qwen3.5-9b-base-pile10k \
  --layers all \
  --preset paper \
  --baselines all \
  --capture-layers-at-once all \
  --output experiments/reconstruction/official/results/qwen35-9b
