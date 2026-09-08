# Fitting–autointerpretability convergence

This experiment asks whether ICA autointerpretability stabilizes as the FastICA
iteration budget increases. Its initial scope is Qwen 3.5 9B Base at residual
layers 15 and 31, using the same one-million-token Pile-10k activation captures
as the official Lens.

`trajectory.py` runs one continuous, deterministic FastICA trajectory per layer.
It stores the initialization, every iteration 1 through 10, and every tenth
iteration from 20 through 200. Checkpoints retain persistent optimization-row
identity and are not independently reordered by final component strength.

Run:

```bash
uv run python experiments/fitting-autointerpretability-convergence/trajectory.py
```

The output is resumable at the latest saved iteration. Shared centers and
whitening matrices are stored once per layer; each checkpoint stores only the
current float32 unmixing matrix. The preparation step extracts only the matched
rows needed by the existing autointerpretability evaluator; it does not
materialize or profile a full Lens at every checkpoint.

Select 50 persistent optimization rows per layer without inspecting their fitted
scores or profiles:

```bash
uv run python experiments/fitting-autointerpretability-convergence/select_cohort.py --dry-run
```

Omit `--dry-run` to write `runs/cohort.json` and calculate consecutive-checkpoint
directional continuity. Each layer uses its layer index as the NumPy RNG seed and
samples rows uniformly without replacement. Tail orientation
remains pending until the selected rows are profiled at iteration 200; that final
orientation will be propagated backward through the sequentially sign-aligned rows.

Prepare matched records at iterations 0, 1, 2, 3, 5, 7, 10, 20, 50, 100, and
200:

```bash
uv run python experiments/fitting-autointerpretability-convergence/select_cohort.py
uv run python experiments/fitting-autointerpretability-convergence/prepare.py --dry-run
uv run python experiments/fitting-autointerpretability-convergence/prepare.py
```

The iteration-200 population-skewness tail is propagated backward through the
sign-aligned trajectory. The 50,000 existing Qwen autointerpretability fragments
then pass through Qwen once for both layers and all checkpoints. The float16
caches are stored under
`~/Expansion/research/ICA-data/fitting-autointerpretability-convergence`; the
evaluator-compatible directories under `runs/prepared` symlink to those caches.

Run the unchanged Cunningham-style evaluator for all prepared checkpoints:

```bash
uv run python experiments/fitting-autointerpretability-convergence/evaluate.py --dry-run
uv run python experiments/fitting-autointerpretability-convergence/evaluate.py
```

Each condition uses the same 50 persistent row IDs. For each row, the existing
protocol explains five top-activating fragments and scores predictions on five
held-out top plus five held-out random fragments. The reported combined score is
Pearson correlation over the resulting 640 token-level activation pairs.
