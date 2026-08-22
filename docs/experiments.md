# Experiments

ICA Lens provides two paper-level experiments. Reconstruction is the primary
test of dictionary quality; sparse probing tests whether a small number of
coordinates carry class-relevant information.

| Experiment | Research question | Main result |
| --- | --- | --- |
| Reconstruction | Can unseen activations be reconstructed from a few reusable directions? | Top-k reconstruction error and cosine similarity |
| Sparse probing | Do a few component coordinates carry concept-relevant information? | Mean probe accuracy against the number of features used |

Both experiments accept a local or published Lens artifact, save
machine-readable results, resume completed work, and create figures in a
separate offline step.

## Reconstruction

### What it measures

A useful dictionary should reconstruct typical, unseen activations accurately
with only a few reusable directions. The reconstruction experiment evaluates
that claim on held-out text.

For every token, each method retains its top-k directions and reconstructs the
hidden state. ICA, a registered public SAE, PCA, and a seeded random orthogonal
basis receive exactly the same held-out tokens. The experiment reports:

- **Cosine similarity**, which measures agreement in direction.
- **Normalized MSE**, which also measures magnitude error relative to a
  mean-prediction baseline.

The current experiment studies **top-k reconstruction**: keep the strongest k
directions and discard the rest. A complementary **top-k ablation
reconstruction** experiment—remove the strongest k directions and reconstruct
from what remains—is planned separately.

### Start with a smoke run

```bash
icalens experiment reconstruction \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --baselines all \
  --output experiments/reconstruction/pilot-runs/gpt2-smoke
```

For the full experiment, use `--layers all --preset paper`. The paper preset
evaluates 1,024-token contexts in six distinct domains: news (AG News), general
prose (WikiText), source code (GitHub Code), Spanish Wikipedia, Chinese
Wikipedia, and multi-turn dialogue (UltraChat). This is a compact diversity
suite rather than an exhaustive sample of language.

For GPT-2, the result also includes a dashed SAE control restricted to token
positions 0–63, matching that SAE's training context. The solid SAE curve and
all primary comparisons use the complete evaluation context.

### Execution and recovery

The experiment is dataset-first. For each dataset, all pending layers are
captured in shared model passes, checkpointed, evaluated sequentially, and then
removed after their results are durable. Repeating the command resumes completed
dataset-layer tasks.

`--capture-layers-at-once` controls the memory/speed tradeoff. It accepts a
positive integer or `all` and defaults to `all`; smaller groups repeat the model
pass but reduce peak memory. Use `--context-length` and
`--max-tokens-per-dataset` for diagnostics. The resolved settings and source
provenance are recorded in `run.json`.

The `pile10k` preset is an in-distribution sanity check on the ICA fitting
corpus. It is useful diagnostically, but it is not a substitute for the held-out
`paper` preset.

### Create figures

```bash
icalens experiment figure reconstruction \
  experiments/reconstruction/pilot-runs/gpt2-smoke
```

The command writes PNG figures and caption text under the experiment's
`figures/` directory. It produces aggregate views and, when the results contain
multiple layers or datasets, subplot figures broken down by layer and by
dataset. Pass several experiment directories to create aligned model panels;
add `--format png,pdf` when PDF output is also needed.

The source repository retains the exact commands, accepted results, and final
paper figures under `experiments/reconstruction/official/`.

## Sparse probing

### What it measures

The sparse-probing experiment runs the paper's SAEBench protocol. For each
classification dataset, SAEBench ranks feature dimensions on the training split,
trains a supervised linear probe using only the top-ranked features, and reports
accuracy on held-out examples. This tests how concentrated class-relevant
information is under a small feature budget.

ICA and PCA coordinates are signed, so each coordinate is exposed as separate
positive and negative nonnegative features. ICA, a registered public SAE, PCA,
and a seeded random orthogonal basis can therefore be evaluated with identical
datasets, splits, feature budgets, and probes.

### Start with a smoke run

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output experiments/sparse-probing/pilot-runs/gpt2-smoke
```

The `smoke` preset is a compatibility check, not a paper result. The `paper`
preset uses the eight paper datasets, full train/test sizes, and standard
feature budgets. To compare all registered methods:

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6,10 \
  --preset paper \
  --baselines all \
  --output experiments/sparse-probing/pilot-runs/gpt2-paper-comparison
```

The packaged registry pins supported public SAE checkpoints and their
preprocessing: GPT-2 OAI v5 ReLU, Gemma Scope JumpReLU, and Qwen Scope TopK
SAEs. SAEBench and requested checkpoints are downloaded lazily. Use `--dry-run`
to inspect the resolved model, backend, datasets, and baselines without running
the benchmark. An existing checkout can be supplied with
`--saebench-path /path/to/SAEBench`.

### Execution and recovery

Sparse probing is also dataset-first. For one dataset, it captures every
requested layer in shared model passes, then evaluates ICA and all requested
baselines before moving to the next dataset. Activation caches are removed only
after the corresponding results are durable. Repeating the same command skips
completed work.

Before starting, ICA Lens estimates the largest activation cache, applies a 20%
safety margin, and checks the output filesystem. Use `--allow-low-disk` only to
deliberately override this safeguard. The terminal panel reports overall
progress, dataset and method indices, elapsed time, and ETA; complete SAEBench
output remains in the experiment's log files.

### Create figures

Figure generation is offline and does not load the model or SAEBench:

```bash
icalens experiment figure sparse-probing \
  experiments/sparse-probing/pilot-runs/gpt2-smoke
```

PNG is the default, written to the run's `figures/` directory with caption text.
Use `--format png,pdf` for both formats, `--force` to replace files, or
`--output PATH` to select another destination. Multiple result directories
create aligned model-comparison panels.

The source repository retains the exact commands, accepted results, and final
paper figure under `experiments/sparse-probing/official/`.

## Reproducibility notes

- Use a clean Git worktree for paper runs. ICA Lens warns when local source
  changes are uncommitted.
- Keep each distinct configuration in its own output directory. An existing
  `run.json` is reused only when its configuration matches.
- Exploratory runs belong under the corresponding experiment's ignored
  `pilot-runs/` directory. Promote only accepted commands, results, and figures
  to its tracked `official/` directory.
