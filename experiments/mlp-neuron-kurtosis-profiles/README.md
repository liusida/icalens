# MLP-neuron kurtosis profiles

This experiment studies the scalar MLP neuron coefficients entering Qwen 3.5
9B Base's `down_proj`.  Neuron `N<i>` is paired with column `i` of
`W_downproj`; the measured scalar is the post-gating coefficient multiplying
that column.

The default run uses zero-based layer 16, the middle of the model's 32 blocks,
and mirrors the official ICA profiling populations:

- population mean, variance, skewness, and excess kurtosis use all 1,000,000
  cached fitting-token positions;
- the dominant tail is positive for nonnegative population skewness and negative
  otherwise; and
- top occurrences are searched in the same seed-0 uniform sample of 100,000
  fitting rows used by ICA component profiling.

After measuring all 12,288 neurons, the script selects three lowest-kurtosis
neurons, three seed-0 random neurons above the median, and three highest-kurtosis
neurons.  It stores 20 occurrences from each neuron's dominant tail.

```bash
uv run python experiments/mlp-neuron-kurtosis-profiles/run.py
```

Once the statistics phase is complete, the distributions can be plotted while
the occurrence search is still pending:

```bash
uv run python experiments/mlp-neuron-kurtosis-profiles/plot.py
```

This writes `figures/neuron-statistics.png`, containing the all-neuron excess
kurtosis and signed-skewness histograms.

After occurrence search completes, render the nine selected neuron profiles as
a self-contained tabbed report:

```bash
uv run python experiments/mlp-neuron-kurtosis-profiles/enrich_contexts.py
uv run python experiments/mlp-neuron-kurtosis-profiles/report.py
```

The enrichment pass revisits only occurrence documents and records 48 tokens
before and 8 tokens after each target, including the selected neuron's score at
every position. The report writes `reports/neuron-profiles.html`; token
backgrounds encode signed scores and an underline marks the target token.

For interactive hypothesis testing, open `neuron_prompt_playground.ipynb` and
edit the final prompt cell. Its `analyze(...)` helper supports any layer-16
neuron, optional underlined token indices, exact-score tooltips, and the same
fixed linear background colors as the HTML report.

Repeating the command resumes from compatible checkpoints in `runs/main`.
Changing any result-defining option requires a different `--output` directory.
During occurrence search, `selected-neuron-profiles.partial.json` is refreshed
every 100 documents by default, and `occurrences.partial.npz` retains the same
state for automatic resume. The partial JSON is replaced by the final
`selected-neuron-profiles.json` when all documents are complete.

