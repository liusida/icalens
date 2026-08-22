#!/usr/bin/env bash
set -euo pipefail

uv run icalens experiment reconstruction \
  --lens sida/icalens-gemma-2-2b-pile10k \
  --layers all \
  --preset paper \
  --baselines all \
  --capture-layers-at-once all \
  --output experiments/reconstruction/official/results/gemma2
