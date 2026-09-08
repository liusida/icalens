# Trajectory-wide fixed-fragment protocol

The checkpoint-specific protocol changes its five explanation fragments and five
held-out top fragments whenever a component's activation ranking changes. That is
appropriate for measuring each checkpoint on its own strongest occurrences, but
it confounds longitudinal score changes with changes in the evaluated text.

The fixed-panel condition isolates longitudinal change. For each persistent ICA
row and layer, it:

1. ranks the 50,000 fragments by their maximum token activation at every evaluated
   fitting checkpoint;
2. takes the union of the top 20 lists;
3. samples ten unique-document fragments without replacement, with selection
   probability proportional to the number of checkpoint top-20 lists containing
   each fragment;
4. assigns the first five sampled fragments to explanation and the remaining five
   to held-out validation; and
5. uses these same two sets at every checkpoint.

Weighting by top-20 membership is equivalent to first drawing a checkpoint
uniformly and then drawing one of its top fragments, while deduplication prevents
stable fragments from appearing multiple times. The deterministic random seed is
derived from the layer and persistent component ID. Sampling occurs before any
explainer or simulator request. Explanation and validation fragments also come
from different source documents.

Five fixed random-validation fragments are sampled outside the top-fragment union,
must activate the component at every checkpoint, and cannot share source documents
with either top split. They preserve the existing evaluator schema, although the
primary fixed-panel statistic is `top_score`.

This condition measures whether a checkpoint's explanation predicts activation on
a common, trajectory-wide collection of salient contexts. It does not replace the
checkpoint-specific condition, which measures intrinsic interpretability on each
checkpoint's own strongest contexts.

The complete four-layer run is built by one command. Existing two-layer
trajectory folders are immutable checkpoint caches: their files are imported
into one unified trajectory using hard links when possible (and atomic copies
otherwise). Hard links allow the old paths to be removed later without doubling
checkpoint storage. If no caches are supplied and no unified trajectory exists,
the same builder fits the four-layer trajectory from scratch.

```bash
uv run python experiments/fitting-autointerpretability-convergence/prepare_fixed_panel_run.py
```

This creates the following self-contained run layout:

```text
runs/fixed-panel/
├── trajectory/
├── cohort.json
├── checkpoint-prepared/
├── prepared/
├── evaluated-tinker/
└── fixed-panel-run.json
```

The large projected activation arrays remain in the external `--archive`; the
two prepared directories contain metadata and relative links. Repeating the
same command validates and resumes the preparation.

Then evaluate it independently:

```bash
uv run python experiments/fitting-autointerpretability-convergence/evaluate.py \
  --layers 7,15,23,31 \
  --input experiments/fitting-autointerpretability-convergence/runs/fixed-panel/prepared \
  --output experiments/fitting-autointerpretability-convergence/runs/fixed-panel/evaluated-tinker
```
