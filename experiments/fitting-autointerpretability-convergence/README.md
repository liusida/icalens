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
Pearson correlation over the resulting 640 token-level activation pairs. Tinker
explanation, simulation, and teacher-forced scoring requests use sampling seed 0
so an identical request is reproducible. The
runner is component-major: it evaluates one matched cohort position across all
11 fitting checkpoints before advancing to the next position. The 11
checkpoints for that component run concurrently, and each checkpoint permits up
to 10 concurrent simulator requests. These limits can be changed with
`--max-concurrent-checkpoints` and `--max-concurrent-simulations`; the defaults
permit up to 110 simultaneous simulator requests during the simulation phase.
Consequently, a partial run already provides matched convergence curves for
every fully completed cohort position. The parent process owns the sole live
terminal display; complete output from each concurrent child is kept in
`runs/evaluated/logs/component-XX-iter-YYY.log` so their displays cannot
overwrite one another.

Plot the currently available individual trajectories and aggregate means
without rerunning evaluation:

```bash
uv run python experiments/fitting-autointerpretability-convergence/plot.py
```

Use `--force` to refresh the figures as more component trajectories finish.
Missing evaluations remain missing rather than being interpolated.

Inspect one persistent component's generated explanation across checkpoints:

```bash
uv run python experiments/fitting-autointerpretability-convergence/inspect_explanations.py \
  --layer 31 --component 3334
```

The default view prints each checkpoint's explanation and the signed cosine
similarity of its residual-space reading direction to iteration 200. Because
the checkpoints belong to one continuous optimization trajectory with
persistent row identity, the sign is retained. Add `--show-scores` when the
numerical autointerpretability scores are also useful.

Use `--input runs/evaluated-tinker-no-seed` (with the full experiment-relative
path) to inspect an archived evaluator condition.

##### Additive layers 7 and 23
>
> Layers 7 and 23 are an additive extension. They use separate run and cache
> roots, so none of the completed layer-15/layer-31 artifacts are invalidated or
> recomputed.

Fit and select the two new trajectories:

```bash
uv run python experiments/fitting-autointerpretability-convergence/trajectory.py \
  --layers 7,23 \
  --output experiments/fitting-autointerpretability-convergence/runs/trajectory-layers-07-23

uv run python experiments/fitting-autointerpretability-convergence/select_cohort.py \
  --layers 7,23 \
  --trajectory experiments/fitting-autointerpretability-convergence/runs/trajectory-layers-07-23 \
  --output experiments/fitting-autointerpretability-convergence/runs/cohort-layers-07-23.json
```

Prepare the same fragment corpus for only the two new layers:

```bash
uv run python experiments/fitting-autointerpretability-convergence/prepare.py \
  --layers 7,23 \
  --trajectory experiments/fitting-autointerpretability-convergence/runs/trajectory-layers-07-23 \
  --cohort experiments/fitting-autointerpretability-convergence/runs/cohort-layers-07-23.json \
  --output experiments/fitting-autointerpretability-convergence/runs/prepared-layers-07-23 \
  --archive /media/liusida/Expansion/research/ICA-data/fitting-autointerpretability-convergence-layers-07-23
```

Evaluate only those layers:

```bash
uv run python experiments/fitting-autointerpretability-convergence/evaluate.py \
  --layers 7,23 \
  --input experiments/fitting-autointerpretability-convergence/runs/prepared-layers-07-23 \
  --output experiments/fitting-autointerpretability-convergence/runs/evaluated-tinker-layers-07-23
```

Plot the extension with the same two-panel convention:

```bash
uv run python experiments/fitting-autointerpretability-convergence/plot.py \
  --layers 7,23 \
  --input experiments/fitting-autointerpretability-convergence/runs/evaluated-tinker-layers-07-23 \
  --preparation experiments/fitting-autointerpretability-convergence/runs/prepared-layers-07-23 \
  --output experiments/fitting-autointerpretability-convergence/figures/layers-07-23
```
