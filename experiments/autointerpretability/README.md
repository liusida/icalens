# Autointerpretability experiment

This experiment will compare how accurately an LLM-generated explanation predicts
held-out ICA-component and SAE-feature activations. It reimplements only the
autointerpretability protocol from Cunningham et al. (2023); it does not train a
language model, ICA Lens, or SAE.

The pilot pipeline is implemented. No paid evaluator calls or official measurements
have been run yet.

## Run the GPT-2 pilot

The Tinker client runs in a pinned environment under
`~/.cache/icalens/environments/`. ICA Lens creates it on the first Tinker run;
its Transformers dependency does not modify the main project environment.

Prepare ten ICA components and ten SAE features at the GPT-2 midpoint layer from a
shared pool of 50,000 OpenWebText fragments:

```bash
uv run icalens experiment autointerpretability prepare \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 5 \
  --n-features 10 \
  --output experiments/autointerpretability/pilot-runs/gpt2-layer5
```

Inspect the number of Tinker requests without making them:

```bash
uv run icalens experiment autointerpretability evaluate \
  --input experiments/autointerpretability/pilot-runs/gpt2-layer5 \
  --output experiments/autointerpretability/pilot-runs/gpt2-layer5-tinker \
  --n-features 10 \
  --dry-run
```

Then run the Tinker condition. Inkling generates each explanation and Qwen3.8-27B
simulates the held-out activations. The command reads `TINKER_API_KEY` from the
project-root `.env` and resumes feature and fragment responses written atomically:

```bash
uv run icalens experiment autointerpretability evaluate \
  --input experiments/autointerpretability/pilot-runs/gpt2-layer5 \
  --output experiments/autointerpretability/pilot-runs/gpt2-layer5-tinker \
  --n-features 10 \
  --max-concurrent 10 \
  --env-file .env
```

The defaults are `thinkingmachines/Inkling` for `--explainer-model` and
`Qwen/Qwen3.8-27B` for `--simulator-model`. After Qwen generates 64 indexed integer
labels, the evaluator makes a teacher-forced Tinker scoring call with
`include_prompt_logprobs=True` and `topk_prompt_logprobs=20`. It converts the
probabilities of labels 0 through 10 into a decimal expected activation and correlates
those predictions with continuous ground-truth activations.

Each feature therefore uses one explanation request, ten simulation generations,
and ten log-probability scoring calls. Features run sequentially; the ten
generate-and-score fragment workflows run concurrently and checkpoint individually.
Lower `--max-concurrent` if the evaluator service applies tighter rate limits.

Create the saved-result figure after evaluation completes:

```bash
uv run icalens experiment figure autointerpretability \
  experiments/autointerpretability/pilot-runs/gpt2-layer5-tinker
```

The command writes PNG, PDF, and a same-stem text companion under the evaluation
directory's `figures/` folder. A single-layer diagnostic shows every finite feature
score together with the method mean and a deterministic feature-level bootstrap 95%
interval. Layer-wise primary figures show only the means and intervals to keep the
comparison legible. Existing files are preserved unless `--force` is supplied.

Compatible evaluation directories for different layers of the same model are merged
into one layer-wise ICA-versus-SAE panel:

```bash
uv run icalens experiment figure autointerpretability \
  experiments/autointerpretability/pilot-runs/gpt2-layer5-tinker \
  experiments/autointerpretability/pilot-runs/gpt2-layers-2-8-11-tinker \
  --force
```

Pass the three completed model evaluations in the conventional GPT-2, Gemma, Qwen
order to create the overall shared-axis figure:

```bash
uv run icalens experiment figure autointerpretability \
  experiments/autointerpretability/runs/gpt2-tinker \
  experiments/autointerpretability/runs/gemma-2-2b-tinker \
  experiments/autointerpretability/runs/qwen3.5-9b-tinker \
  --output experiments/autointerpretability/figures \
  --force
```

## Cunningham-modern OpenAI evaluation

Reuse the prepared fragments and feature activations while keeping the Inkling
condition intact:

```bash
uv run --extra autointerpretability icalens experiment autointerpretability evaluate \
  --provider openai \
  --input experiments/autointerpretability/pilot-runs/gpt2-layer5 \
  --output experiments/autointerpretability/pilot-runs/gpt2-layer5-openai-modern \
  --n-features 10
```

This path follows `modern_interpret.py`: strict 64-label structured output,
`top_logprobs=20`, probability-weighted expected labels, continuous ground-truth
activations, ten concurrent simulator calls, one-second request pacing, five retries,
and NaN-preserving condition aggregation. ICA orientation and uniform eligible-feature
selection remain fixed by preparation rather than inferred from evaluation data.

## Research question

For a fixed language model and residual-stream layer, are features from an existing
ICA Lens at least as predictable from a natural-language explanation as features
from an existing pretrained SAE?

The primary per-feature metric is the Pearson correlation between the simulator's
predicted token activations and the true token activations on held-out top-activating
and random text fragments. We call this the **top-and-random autointerpretability
score**, following the paper.

## Scope

The initial comparison will cover:

| Model | ICA Lens | Pretrained SAE | Initial layers |
| --- | --- | --- | --- |
| GPT-2 small | `sida/icalens-gpt2-small-pile10k` | GPT2-Small OAI v5 32k | 2, 5, 8, 11 |
| Gemma 2 2B | `sida/icalens-gemma-2-2b-pile10k` | Gemma Scope 2B residual 16k | 5, 12, 18, 25 |
| Qwen 3.5 9B Base | `sida/icalens-qwen3.5-9b-base-pile10k` | Qwen Scope residual 64k TopK-50 | 7, 15, 23, 31 |

These are the layers nearest the 25%, 50%, 75%, and 100% depth marks under the
project's zero-based post-block layer convention. All model, Lens, SAE, dataset, and
evaluator revisions must be pinned in the run manifest. The SAE checkpoint
definitions come from `src/icalens/experiments/baseline_registry.json`.

The first official run should use these four depth-matched layers per model rather
than every layer. At 150 features per method, the complete matrix contains 3,600
feature-method evaluations. We can extend to all layers after inspecting the pilot
and official cost report.

## Protocol

### 1. Build a shared text pool

- Use the pinned `Skylion007/openwebtext` repository and train split to remain close
  to the original experiment.
- Deterministically sample 50,000 documents with one 64-token fragment per document.
- Tokenize separately with each target model's tokenizer. ICA and SAE for the same
  model consume exactly the same fragments and residual activations.
- Reject fragments containing tokenizer replacement characters, and record every
  rejection and sampling seed.
- Cache token IDs and the selected candidate-feature activations by model and layer
  so evaluator reruns do not repeat model inference.

The fragment-pool size does not affect evaluator-call count. Every evaluated feature
uses one explanation and ten simulation generations; Tinker additionally uses ten
teacher-forced scoring calls. A larger pool increases only local tokenization, model
inference, ICA/SAE encoding, preparation time, and candidate-activation storage. In
return, it provides better top-activating examples and substantially reduces
rare-feature eligibility bias.

### 2. Select features without favoring either method

The original paper evaluated the first 150 features. That is unsuitable here:
current ICA component IDs are ordered by logcosh deviation, whereas SAE IDs have a
different checkpoint-specific ordering. Taking the first IDs would deliberately
select unusually non-Gaussian ICA components.

Instead, for every model, layer, and method:

1. create a fixed random permutation of all feature IDs from a recorded seed;
2. scan IDs in that order;
3. retain the first 150 features that have at least 20 top records and 20 nonzero
   random records in the shared pool;
4. record all attempted, accepted, and rejected IDs and rejection reasons.

This samples uniformly from eligible features within each dictionary. ICA and SAE
features cannot be paired by ID, so method confidence intervals and comparisons are
across independent feature samples. The report will also show eligibility rates for
the full attempted sample, including dead and insufficiently active features, so the
filter cannot hide a method's failure to produce evaluable features.

### 3. Put ICA and SAE activations on the same one-sided contract

SAE activations are their checkpoint-defined nonnegative encoder activations.

For ICA component `c`, use the tail direction stored by profiling, chosen from the
profiling corpus rather than the evaluation pool:

```text
oriented_activation = max(0, tail_direction[c] * score[c])
```

This removes the arbitrary ICA sign while retaining the original protocol's
nonnegative feature-activation assumption. The run must fail if the selected ICA
profile or tail-direction provenance is missing; it must never infer the sign from
evaluation examples. We will preserve raw signed scores in diagnostic artifacts,
but they are not shown to either evaluator LLM.

### 4. Construct records and splits

For each eligible feature:

- rank all fragments by their maximum token activation and retain the top 20;
- draw 20 activating random fragments using a feature-specific deterministic seed;
- preserve the original interleaved split convention;
- use five top records for explanation generation;
- score on five held-out top records and five held-out random records;
- retain the remaining records as calibration/test material for protocol
  compatibility and future checks, but do not silently mix them into the primary
  score.

The five explanation examples are clipped below at zero and discretized to integers
0 through 10 using the historical maximum-activation normalization. Scoring retains
the held-out ground-truth activations as continuous values. Exact selected record IDs
are stored per feature.

### 5. Use separate explainer and simulator roles

- The **explainer LLM** receives only the five training fragments, their tokens, and
  normalized activations, and returns one concise feature explanation.
- The **simulator LLM** receives that explanation and one held-out fragment at a
  time. Tokens are explicitly indexed from 0 through 63. It generates exactly 64
  integer labels in `[0, 10]`; Tinker uses an indexed object and OpenAI uses a strict
  fixed-length array.
- The simulator never receives held-out true activations.
- The Tinker condition uses `thinkingmachines/Inkling` as explainer and
  `Qwen/Qwen3.8-27B` as simulator, with both exact model IDs recorded in the run
  configuration.
- Prompts and structured-output contracts are versioned in code, and their combined
  hash is stored in `run.json` and every reusable evaluator checkpoint.

For Tinker, the indexed object lets the validator detect missing, duplicate, extra,
or out-of-order indices rather than merely counting numbers. Failed structure is
retried and then recorded as an evaluator failure; values are never guessed or
padded. The generated labels are only anchors: label probabilities produce the
continuous expected predictions used for correlation.

The project-root `.env` already provides `TINKER_API_KEY` and `OPENAI_API_KEY`.
Evaluator clients load these names at runtime without copying their values into
configuration, logs, cached responses, or result manifests.

For a feature, concatenate predictions and truths over the five top and five random
fragments (640 token positions) and compute Pearson correlation. Also retain top-only
and random-only correlations as diagnostics. Constant or invalid vectors produce an
explicit undefined result; they are never silently dropped.

### 6. Aggregate and report

For each model, layer, and method, report:

- mean top-and-random correlation with a 95% feature-level bootstrap interval;
- the full per-feature score distribution;
- top-only and random-only diagnostic means;
- completion, evaluator failure, invalid-output, and undefined-correlation counts;
- selected/rejected feature counts and reasons;
- captured-token, local-compute, API-token, latency, and estimated-cost totals.

The primary figure compares ICA and SAE within each model and layer. It must follow
`notes/plot-style-policy.md`, use committed result files only, and export PNG, PDF,
and the source-values text companion.

## Reproducibility and resume contract

The implementation will follow `notes/long-run-policy.md`:

- repeating the same command and output path validates and resumes automatically;
- incompatible configuration or provenance fails before model/API work;
- dataset fragments, captured layers, feature records, explanations, and individual
  simulations are atomic durable units;
- completed API responses are cached by prompt hash and never purchased twice;
- errors are durable diagnostic records; rerunning retries only incomplete work and
  preserves valid checkpoints;
- complete logs live below the experiment output, and dirty source emits the shared
  visible warning.

Before paid execution, a dry run must print the number of feature explanations,
simulation calls, estimated input/output tokens, and a conservative cost bound. A
configured budget ceiling stops dispatch before it can be exceeded.

## Repository layout

```text
experiments/autointerpretability/
├── README.md
├── official/
│   ├── README.md
│   ├── config/
│   ├── scripts/
│   ├── results/
│   └── figures/
└── pilot-runs/                 # ignored local runs

src/icalens/experiments/
├── _display.py                         # reusable compact progress and full logging
├── _run.py                             # reusable run validation and atomic metadata
├── autointerpretability.py              # preparation, Tinker evaluation, summary
└── autointerpretability_protocol.py     # prompts, splits, validation, scoring
```

Only this experiment uses the reusable run/display framework for now. Existing
experiments retain their current runners until they are migrated separately.

The implementation reuses the current pretrained-SAE loader rather than duplicating
checkpoint logic. Activation capture reuses the existing Hugging Face
residual-stream and GB10 model-loading paths, including Qwen 3.5 support.

## Implementation status and remaining stages

1. **Pure protocol core — implemented.** Record selection, splitting, normalization,
   prompt rendering, response validation, Pearson scoring, and aggregation with
   deterministic unit tests and synthetic fixtures.
2. **Feature adapters — implemented.** Reuse the ICA Lens transform and the pinned
   pretrained-SAE encoder used by the existing experiment infrastructure. Output
   equivalence tests against saved reconstruction/sparse-probing activations remain
   part of protocol validation.
3. **Capture and record cache — implemented.** Stream the 50,000 fragments, share
   multi-layer capture, retain only selected-feature values, and atomically write
   provenance.
4. **Evaluator backend — implemented for Tinker/Inkling.** It has role-separated
   prompts, indexed structured outputs, retries, response caching, and a request-count
   dry run. Cost estimation, a hard monetary budget, and bounded concurrency remain
   necessary before using a paid provider.
5. **Pilot — ready to run.** Run one layer of GPT-2 with 10 features per method,
   using Inkling as both explainer and simulator through Tinker. Manually inspect
   selected records, explanations, indexed simulator output, resume behavior,
   invalid outputs, eligibility coverage, and cost before scaling.
6. **Protocol validation.** On the local Cunningham reproduction, compare record
   splits, normalized labels, and correlations for a small fixed fixture. Exact
   score equality is not expected when evaluator models differ, but all deterministic
   preprocessing and scoring should agree.
7. **Official run.** Freeze configs and prompts, evaluate 150 features per method at
   the twelve model/layer pairs, then generate figures from committed results.

## Decisions to freeze after the pilot

- whether Inkling remains both the explainer and simulator or the official run uses
  distinct pinned evaluator models;
- evaluator temperature/seed support and whether to repeat generations;
- OpenWebText repository revision and deterministic document-selection rule;
- feature eligibility threshold and whether the sensitivity analysis warrants a
  different primary population;
- bootstrap seed and number of resamples;
- official API budget ceiling.

These choices must be frozen before looking at the full ICA-versus-SAE result.

## References

- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in
  Language Models*, Section 3 and Appendix A: <https://arxiv.org/abs/2309.08600>
- Bills et al., *Language models can explain neurons in language models*:
  <https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html>
- Local reference implementation:
  `/home/liusida/research/ICA-paper/Cunningham-SAE-paper-baseline/Cunningham_2023_sparse_coding/`
