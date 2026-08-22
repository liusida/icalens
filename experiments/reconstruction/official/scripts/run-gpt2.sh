#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment reconstruction \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers all \
  --preset paper \
  --baselines all \
  --capture-layers-at-once all \
  --output experiments/reconstruction/official/results/gpt2
