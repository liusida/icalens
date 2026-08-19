# Experiments

ICA Lens can run the paper's SAEBench sparse-probing protocol from a published
or local Lens artifact. The benchmark is not downloaded during installation.
On first use, the command selects the model-compatible repository and exact
commit, then prepares it in a managed cache.

## GPT-2 smoke run

Use one layer and a small dataset split to verify the complete path:

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output results/gpt2-smoke
```

The smoke preset is a compatibility check, not a paper result. Use
`--preset paper` for the paper's eight datasets, train/test sizes, and feature
budgets.

The sparse-probing adapter splits every signed ICA score into two nonnegative
features, so the positive and negative sides can be selected independently.
Completed layers are checkpointed and reused when the same command resumes.

## Compare with SAE, PCA, and a random basis

GPT-2, Gemma 2 2B, and Qwen3.5 9B Base have registered public-SAE and PCA
baselines. A seeded full-rank random orthogonal basis is available for every
model. Run all four methods through the same datasets, splits, feature
budgets, and SAEBench probes:

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6,10 \
  --preset paper \
  --baselines sae,pca,random \
  --output results/gpt2-paper-comparison
```

The registered SAEs are GPT-2 OAI v5 ReLU, Gemma Scope JumpReLU, and Qwen Scope
TopK-50 checkpoints. The Qwen3.5 9B release has width 65,536 and each selected
layer downloads a checkpoint of about 2.15 GB. PCA uses the same L2-normalized,
centered fitting space as the ICA Lens; because the Lens is full-rank, its saved
whitening transform is sufficient to recover the fitted PCA basis without
collecting a second fitting corpus. Signed ICA and PCA coordinates are both
split into positive and negative features.

Within each layer, SAEBench captures raw model activations once, temporarily
saves them, and reuses the same cache for ICA, SAE, PCA, and Random. The cache
is deleted after every requested representation finishes. Resume operates at
the `(layer, method)` level, so adding a baseline runs only missing methods.

Datasets run sequentially: each dataset is captured, evaluated by every requested
method, and deleted before the next dataset starts. ICA Lens estimates the
largest single-dataset cache and compares it with free space on the output
filesystem. The check includes a 20% safety margin and stops early when space is
insufficient. Use `--allow-low-disk` only to deliberately override this
safeguard. With the paper preset in float32, the peak raw cache is approximately
4.6 GiB for GPT-2 Small, 13.7 GiB for Gemma 2 2B, and 24.4 GiB for Qwen3.5 9B
Base.

During evaluation, the terminal keeps a compact live panel with the overall
dataset-method percentage, current task, elapsed time, and only the latest few
SAEBench messages. The complete unabridged output is retained as
`layers/layer_XX/saebench-detail.log`.

Baseline definitions—including checkpoint revision, layer naming, width, and
preprocessing—live in the packaged baseline registry. A requested public SAE
or PCA baseline is rejected unless it is explicitly registered for the model. Public
checkpoints and SAEBench itself are downloaded only when the experiment runs.

Inspect dependency resolution without downloading or running anything:

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output results/gpt2-smoke \
  --dry-run
```

Use `--saebench-path /path/to/SAEBench` to test an existing checkout. Only
explicitly registered and validated model/backend pairs are accepted.

## Create a paper figure

Figure generation is offline: it does not load a language model or SAEBench.

```bash
icalens experiment figure sparse-probing results/gpt2-smoke
```

Use `--format png,pdf` to write both formats and `--force` to replace existing
files. By default, this command keeps temporary outputs with the experiment at
`results/gpt2-smoke/figures/`. Comparison runs produce one curve per method;
multi-layer results are averaged by feature budget.

When a figure is ready to keep in the repository, publish it explicitly:

```bash
icalens experiment figure sparse-probing results/gpt2-smoke --output figures
```

## Held-out reconstruction

The reconstruction experiment asks whether an unseen activation can be rebuilt
accurately from only a few reusable dictionary directions. It evaluates ICA,
the registered public SAE, PCA, and a seeded random orthogonal basis on exactly
the same held-out tokens. Directions are ranked per token by the norm of their
individual reconstruction contribution. Normalized MSE is the primary metric;
cosine similarity is reported as a complementary directional metric.

Start with the GPT-2 compatibility run:

```bash
icalens experiment reconstruction \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --baselines all \
  --output results/reconstruction-gpt2-smoke
```

For the full held-out suite, use `--layers all --preset paper`. The paper preset
uses six deliberately different domains: news (AG News), encyclopedic prose
(WikiText), source code (GitHub Code), Spanish Wikipedia, Chinese Wikipedia,
and multi-turn dialogue (UltraChat). This is a compact diversity suite rather
than an exhaustive sample of language. Conversations are rendered with common
`User:` and `Assistant:` labels so every model receives identical text.

The paper preset evaluates 1,024-token contexts. For GPT-2, it also reports a dashed SAE control
using only token positions 0–63, matching that SAE's training context; the
solid SAE curve and every primary comparison still use all positions.

Execution is dataset-first. For each dataset, ICA Lens captures every pending
layer in a shared model pass, checkpoints those activations, evaluates layers
sequentially, and removes the dataset cache after every result is durable.
Rerunning the same command resumes completed dataset-layer tasks. The compact
terminal panel reports the current dataset and layer, overall progress, elapsed
time, and ETA. `--capture-layers-at-once` controls the memory/speed tradeoff,
accepts a positive integer or `all`, and defaults to `all`; smaller groups repeat
the model pass but lower peak memory. Evaluation of wide SAE dictionaries is
internally batched. Use `--context-length` to override the preset when running a
diagnostic; the chosen value is recorded in `run.json`.

The `pile10k` preset is an explicitly in-distribution diagnostic on the ICA
fitting corpus. It is useful for comparison with earlier reconstruction runs,
but it is not a replacement for the held-out `paper` preset.

Create both NMSE and cosine figures offline:

```bash
icalens experiment figure reconstruction \
  results/reconstruction-gpt2-smoke
```

Pass several experiment directories to create aligned model panels. PNG and
PDF files plus concise caption text are written under the first experiment's
`figures/` directory by default.
